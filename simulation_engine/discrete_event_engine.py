#!/usr/bin/env python3
import random
import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from stochastic.sample_distribution import SampleDistribution
from multi_agent_system.agent import create_agent, GenericAgent
from multi_agent_system.automaton import create_automaton

@dataclass(order=True)
class Event:
    time: float
    event_type: str
    agent_id: str = field(compare=False)
    target_id: str = field(compare=False)

class DiscreteEventSimulator:
    def __init__(self, agents_config, automata_config, distributions_config, world_config):
        self.agents_config = agents_config
        self.automata_config = automata_config
        self.world_config = world_config
        
        self.sampler = SampleDistribution(distributions_config)
        
        self.population = {}
        self.malicious_agents = {}
        self.transactions = []
        self.event_queue = []
        self.current_time = 0.0
    
    def _create_agent(self, agent_type: str, agent_id: str) -> GenericAgent:
        """Create an agent using the generic factory."""
        return create_agent(agent_type, agent_id, self.agents_config, 
                           self.automata_config, self.sampler)
    
    def _create_population(self):
        """Create all agents based on world_config."""
        agent_counts = self.world_config.get('agents', {})
        
        for agent_type, count in agent_counts.items():
            id_prefix = self.agents_config.get(agent_type, {}).get('id_prefix', f"{agent_type}_")
            
            for i in range(count):
                agent_id = f"{id_prefix}{i+1:06d}"
                agent = self._create_agent(agent_type, agent_id)
                self.population[agent_id] = agent
                
                if agent_type == 'malicious':
                    self.malicious_agents[agent_id] = agent
    
    def schedule_contact_attempt(self, malicious_id: str, citizen_id: str, delay: float):
        """Schedule a contact attempt event."""
        event = Event(
            time=self.current_time + delay,
            event_type='CONTACT_ATTEMPT',
            agent_id=malicious_id,
            target_id=citizen_id
        )
        heapq.heappush(self.event_queue, event)
    
    def schedule_recruitment_attempt(self, malicious_id: str, citizen_id: str, delay: float):
        """Schedule a recruitment attempt event."""
        event = Event(
            time=self.current_time + delay,
            event_type='RECRUITMENT_ATTEMPT',
            agent_id=malicious_id,
            target_id=citizen_id
        )
        heapq.heappush(self.event_queue, event)
    
    def get_random_citizen(self):
        """Get a random citizen from the population."""
        citizens = [c for c in self.population.values() if c.type == 'citizen']
        return random.choice(citizens) if citizens else None
    
    def handle_contact_attempt(self, event: Event):
      """Handle a contact attempt event."""
      malicious = self.population.get(event.agent_id)
      citizen = self.population.get(event.target_id)
      
      if not malicious or not citizen:
          return
      
      # Schedule next contact for this malicious agent
      next_citizen = self.get_random_citizen()
      if next_citizen:
          dist_config = self.sampler.get_distribution('ig_tiktok_daily_contact')
          delay = self.sampler.sample(dist_config) if dist_config else 1.0
          self.schedule_contact_attempt(malicious.id, next_citizen.id, delay)
      
      # Process current contact - CITIZEN decides
      if malicious.state == 'idle' and citizen.state == 'idle':
          local_view = {'age': citizen.get_attribute('age', 0)}
          intention = citizen.step(local_view)  # ← CHANGE HERE: citizen, not malicious
          
          if intention == 'CONTACT_ACCEPTED':
              citizen.update_state('contacted')
              malicious.update_state('recruiting')
              citizen.attributes['last_contact_time'] = self.current_time
              self.schedule_recruitment_attempt(malicious.id, citizen.id, 1.0)
    
    def handle_recruitment_attempt(self, event: Event):
      """Handle a recruitment attempt event."""
      malicious = self.population.get(event.agent_id)
      citizen = self.population.get(event.target_id)
      
      if not malicious or not citizen:
          return
      
      if malicious.state == 'recruiting' and citizen.state == 'contacted':
          local_view = {'age': citizen.get_attribute('age', 0)}
          
          # Get the recruitment automaton (second one)
          automata_names = citizen.get_attribute('automata_names', [])
          if len(automata_names) >= 2:
              # Create or get the recruitment automaton
              recruitment_automaton = create_automaton(automata_names[1], self.automata_config, self.sampler)
              intention = recruitment_automaton.step(local_view)
          else:
              # Fallback to the same automaton
              intention = citizen.step(local_view)
          
          if intention == 'ACCEPT_RECRUITMENT':
              citizen.update_state('active')
              citizen.attributes['mule_count'] = citizen.attributes.get('mule_count', 0) + 1
              malicious.update_state('idle')
              
              self.transactions.append({
                  'time': self.current_time,
                  'malicious_id': malicious.id,
                  'citizen_id': citizen.id,
                  'amount': citizen.get_attribute('income', 0) * 0.1
              })
          else:
              malicious.update_state('idle')

    def initialize_events(self):
        """Schedule initial contact attempts - one per malicious agent."""
        dist_config = self.sampler.get_distribution('ig_tiktok_daily_contact')
        
        for malicious in self.malicious_agents.values():
            target = self.get_random_citizen()
            if target:
                delay = self.sampler.sample(dist_config) if dist_config else 1.0
                self.schedule_contact_attempt(malicious.id, target.id, delay)
    
    def run_simulation(self):
        """Run discrete event simulation."""
        num_citizens = self.world_config.get('agents', {}).get('citizen', 0)
        num_malicious = self.world_config.get('agents', {}).get('malicious', 0)
        max_time = self.world_config.get('max_time', 10000.0)
        
        print("\n" + "="*60)
        print(f"🌱 Creating {num_citizens} citizens and {num_malicious} malicious agents...")
        self._create_population()
        
        print(f"⚙️ Initializing events...")
        self.initialize_events()
        
        print(f"⏰ Running simulation until time {max_time}...")
        events_processed = 0
        
        while self.event_queue and self.current_time <= max_time:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            
            if event.event_type == 'CONTACT_ATTEMPT':
                self.handle_contact_attempt(event)
            elif event.event_type == 'RECRUITMENT_ATTEMPT':
                self.handle_recruitment_attempt(event)
            
            events_processed += 1
            
            if events_processed % 1000 == 0:
                print(f"   Time: {self.current_time:.1f}, Events: {events_processed}")
        
        print(f"✅ Simulation complete. {events_processed} events processed.")
        return self.get_statistics()
    
    def get_statistics(self):
        """Calculate simulation statistics."""
        citizens = [a for a in self.population.values() if a.type == 'citizen']
        active_citizens = [c for c in citizens if c.state == 'active']
        
        # Helper functions
        def get_age(citizen):
            age = citizen.get_attribute('age')
            return age if age is not None else 0
        
        def get_education(citizen):
            edu = citizen.get_attribute('education')
            return edu if edu is not None else 0.5
        
        EDU_THRESHOLD = 0.5
        
        young_low_edu = [c for c in citizens if get_age(c) <= 30 and get_education(c) < EDU_THRESHOLD]
        young_high_edu = [c for c in citizens if get_age(c) <= 30 and get_education(c) >= EDU_THRESHOLD]
        old_low_edu = [c for c in citizens if get_age(c) > 30 and get_education(c) < EDU_THRESHOLD]
        old_high_edu = [c for c in citizens if get_age(c) > 30 and get_education(c) >= EDU_THRESHOLD]
        
        young_low_rate = len([c for c in young_low_edu if c.state == 'active']) / len(young_low_edu) if young_low_edu else 0
        young_high_rate = len([c for c in young_high_edu if c.state == 'active']) / len(young_high_edu) if young_high_edu else 0
        old_low_rate = len([c for c in old_low_edu if c.state == 'active']) / len(old_low_edu) if old_low_edu else 0
        old_high_rate = len([c for c in old_high_edu if c.state == 'active']) / len(old_high_edu) if old_high_edu else 0
        
        stats = {
            'total_citizens': len(citizens),
            'active_mules': len(active_citizens),
            'activation_rate': len(active_citizens) / len(citizens) if citizens else 0,
            'total_transactions': len(self.transactions),
            'avg_mule_count': np.mean([c.get_attribute('mule_count', 0) for c in citizens]) if citizens else 0,
            'final_time': self.current_time,
            'events_processed': len(self.event_queue) + len(self.transactions) * 2,
            'education_threshold': EDU_THRESHOLD
        }
        
        stats['by_profile'] = {
            'young_low_education': young_low_rate,
            'young_high_education': young_high_rate,
            'old_low_education': old_low_rate,
            'old_high_education': old_high_rate,
        }
        
        stats['avg_education_young'] = np.mean([get_education(c) for c in citizens if get_age(c) <= 30]) if citizens else 0
        stats['avg_education_old'] = np.mean([get_education(c) for c in citizens if get_age(c) > 30]) if citizens else 0
        
        return stats

    def print_report(self, stats):
        """Print simulation report."""
        print("\n" + "="*60)
        print("📊 DISCRETE EVENT SIMULATION REPORT")
        print("="*60)
        
        print(f"\n⏰ TIME:")
        print(f"   Final time: {stats['final_time']:.2f}")
        print(f"   Events processed: {stats['events_processed']}")
        
        print(f"\n👥 POPULATION:")
        print(f"   Total citizens: {stats['total_citizens']}")
        print(f"   Active mules: {stats['active_mules']}")
        print(f"   Activation rate: {stats['activation_rate']:.2%}")
        
        print(f"\n💰 TRANSACTIONS:")
        print(f"   Total: {stats['total_transactions']}")
        print(f"   Average per mule: {stats['avg_mule_count']:.2f}")
        
        print(f"\n📈 ACTIVATION RATE BY PROFILE:")
        for profile, rate in stats['by_profile'].items():
            print(f"   {profile}: {rate:.2%}")
        
        print("\n" + "="*60)