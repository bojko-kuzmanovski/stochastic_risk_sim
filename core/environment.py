from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class Profile:
    """Agent profile information."""
    agent_id: str
    is_private: bool


@dataclass
class Follow:
    """Follow relationship between agents."""
    from_agent_id: str
    to_agent_id: str
    is_approved: bool


@dataclass
class Connection:
    """Undirected connection between agents."""
    from_agent_id: str
    to_agent_id: str
    is_approved: bool


@dataclass
class GroupMember:
    """Member of a group."""
    member_agent_id: str
    is_admin: bool


@dataclass
class Group:
    """Group of agents."""
    group_id: str
    members: List[GroupMember]


@dataclass
class DirectChannel:
    """Direct channel between exactly two agents."""
    channel_id: str
    members: List[str]  # exactly 2 members


@dataclass
class GroupChannel:
    """Channel for a group of agents."""
    channel_id: str
    members: List[str]  # member_agent_ids


@dataclass
class Application:
    """Job or opportunity application."""
    publisher_agent_id: str
    applicants: List[str]  # applicant_agent_ids


class Environment:
    """
    Environment containing agents and their relationships.
    Internal class - should not be used directly from outside.
    """
    
    def __init__(self, env_id: str, config: Dict[str, Any], agents):
        """
        Initialize an environment from configuration.
        
        Args:
            env_id: Unique identifier for this environment
            config: Environment configuration (type, modules, constraints)
            agents: Agents instance containing all agents
        """
        self._id = env_id
        self._config = config
        self._agents = agents
        self._modules = config.get('modules', [])
        self._constraints = config.get('constraints', [])
        
        # Internal data structures
        self._context: List[str] = []
        self._profiles: Dict[str, Profile] = {}
        self._follows: Dict[Tuple[str, str], Follow] = {}
        self._connections: Dict[Tuple[str, str], Connection] = {}
        self._groups: Dict[str, Group] = {}
        self._direct_channels: Dict[str, DirectChannel] = {}
        self._group_channels: Dict[str, GroupChannel] = {}
        self._applications: Dict[str, Application] = {}
        
        # Validate and build environment
        self._validate_constraints()
        self._build()
    
    def _validate_constraints(self) -> None:
        """Validate that all constraints are satisfied."""
        for constraint in self._constraints:
            if constraint == "follow requires profile":
                if "follow" in self._modules and "profile" not in self._modules:
                    raise ValueError(f"Environment '{self._id}': follow requires profile")
            elif constraint == "direct_channels requires follow":
                if "direct_channels" in self._modules and "follow" not in self._modules:
                    raise ValueError(f"Environment '{self._id}': direct_channels requires follow")
            elif constraint == "direct_channels requires connection":
                if "direct_channels" in self._modules and "connection" not in self._modules:
                    raise ValueError(f"Environment '{self._id}': direct_channels requires connection")
            elif constraint == "connection requires profile":
                if "connection" in self._modules and "profile" not in self._modules:
                    raise ValueError(f"Environment '{self._id}': connection requires profile")
            elif constraint == "group_channels requires connection":
                if "group_channels" in self._modules and "connection" not in self._modules:
                    raise ValueError(f"Environment '{self._id}': group_channels requires connection")
            elif constraint == "applications requires profile":
                if "applications" in self._modules and "profile" not in self._modules:
                    raise ValueError(f"Environment '{self._id}': applications requires profile")
    
    def _build(self) -> None:
        """Build the environment structure based on modules."""
        # Profile module
        if "profile" in self._modules:
            self._build_profiles()
        
        # Follow module
        if "follow" in self._modules:
            self._build_follows()
        
        # Connection module
        if "connection" in self._modules:
            self._build_connections()
        
        # Groups module
        if "groups" in self._modules:
            self._build_groups()
        
        # Direct channels module
        if "direct_channels" in self._modules:
            self._build_direct_channels()
        
        # Group channels module
        if "group_channels" in self._modules:
            self._build_group_channels()
        
        # Applications module
        if "applications" in self._modules:
            self._build_applications()
    
    def _build_profiles(self) -> None:
        """Build profiles for all agents."""
        for agent in self._agents.get_all_agents():
            self._profiles[agent.id] = Profile(agent_id=agent.id, is_private=False)
    
    def _build_follows(self) -> None:
        """Build follow relationships (simplified - random for now)."""
        import random
        agents = self._agents.get_all_agents()
        for from_agent in agents:
            for to_agent in agents:
                if from_agent.id != to_agent.id and random.random() < 0.1:
                    key = (from_agent.id, to_agent.id)
                    self._follows[key] = Follow(
                        from_agent_id=from_agent.id,
                        to_agent_id=to_agent.id,
                        is_approved=True
                    )
    
    def _build_connections(self) -> None:
        """Build connection relationships (simplified - random for now)."""
        import random
        agents = self._agents.get_all_agents()
        for i, agent_a in enumerate(agents):
            for agent_b in agents[i+1:]:
                if random.random() < 0.05:
                    key = tuple(sorted([agent_a.id, agent_b.id]))
                    self._connections[key] = Connection(
                        from_agent_id=agent_a.id,
                        to_agent_id=agent_b.id,
                        is_approved=True
                    )
    
    def _build_groups(self) -> None:
        """Build groups (simplified - one group with all agents)."""
        import random
        agents = self._agents.get_all_agents()
        if agents:
            group_id = f"{self._id}_default_group"
            members = []
            for i, agent in enumerate(agents):
                members.append(GroupMember(
                    member_agent_id=agent.id,
                    is_admin=(i == 0)
                ))
            self._groups[group_id] = Group(group_id=group_id, members=members)
    
    def _build_direct_channels(self) -> None:
        """Build direct channels between connected agents."""
        channel_counter = 0
        for (a, b), conn in self._connections.items():
            if conn.is_approved:
                channel_id = f"{self._id}_direct_channel_{channel_counter}"
                self._direct_channels[channel_id] = DirectChannel(
                    channel_id=channel_id,
                    members=[a, b]
                )
                channel_counter += 1
    
    def _build_group_channels(self) -> None:
        """Build group channels for each group."""
        for group_id, group in self._groups.items():
            channel_id = f"{self._id}_group_channel_{group_id}"
            member_ids = [m.member_agent_id for m in group.members]
            self._group_channels[channel_id] = GroupChannel(
                channel_id=channel_id,
                members=member_ids
            )
    
    def _build_applications(self) -> None:
        """Build applications (simplified - one application per agent)."""
        for agent in self._agents.get_all_agents():
            self._applications[agent.id] = Application(
                publisher_agent_id=agent.id,
                applicants=[]
            )
    
    # Public methods for agent interactions
    
    def get_profile(self, agent_id: str) -> Optional[Profile]:
        """Get profile for an agent."""
        return self._profiles.get(agent_id)
    
    def get_follows(self, from_agent_id: str) -> List[Follow]:
        """Get all follows from an agent."""
        return [f for (frm, _), f in self._follows.items() if frm == from_agent_id]
    
    def get_followers(self, to_agent_id: str) -> List[Follow]:
        """Get all followers of an agent."""
        return [f for (_, to), f in self._follows.items() if to == to_agent_id]
    
    def are_connected(self, agent_a: str, agent_b: str) -> bool:
        """Check if two agents are connected."""
        key = tuple(sorted([agent_a, agent_b]))
        conn = self._connections.get(key)
        return conn is not None and conn.is_approved
    
    def get_connections(self, agent_id: str) -> List[str]:
        """Get all agents connected to the given agent."""
        connected = []
        for (a, b), conn in self._connections.items():
            if conn.is_approved:
                if a == agent_id:
                    connected.append(b)
                elif b == agent_id:
                    connected.append(a)
        return connected
    
    def get_group_members(self, group_id: str) -> List[str]:
        """Get all member IDs of a group."""
        group = self._groups.get(group_id)
        if not group:
            return []
        return [m.member_agent_id for m in group.members]
    
    def get_direct_channel_members(self, channel_id: str) -> List[str]:
        """Get members of a direct channel."""
        channel = self._direct_channels.get(channel_id)
        return channel.members if channel else []
    
    def get_group_channel_members(self, channel_id: str) -> List[str]:
        """Get members of a group channel."""
        channel = self._group_channels.get(channel_id)
        return channel.members if channel else []
    
    def add_application(self, publisher_id: str, applicant_id: str) -> None:
        """Add an applicant to a publisher's application."""
        if publisher_id not in self._applications:
            raise ValueError(f"Publisher '{publisher_id}' not found")
        if applicant_id not in self._agents:
            raise ValueError(f"Applicant '{applicant_id}' not found")
        
        app = self._applications[publisher_id]
        if applicant_id not in app.applicants:
            app.applicants.append(applicant_id)
    
    def get_applicants(self, publisher_id: str) -> List[str]:
        """Get all applicants for a publisher."""
        app = self._applications.get(publisher_id)
        return app.applicants if app else []
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def modules(self) -> List[str]:
        return self._modules.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert environment to dictionary for serialization."""
        return {
            'id': self._id,
            'modules': self._modules,
            'profile': [{'agent_id': p.agent_id, 'is_private': p.is_private} 
                       for p in self._profiles.values()],
            'follow': [{'from_agent_id': f.from_agent_id, 
                       'to_agent_id': f.to_agent_id, 
                       'is_approved': f.is_approved} 
                      for f in self._follows.values()],
            'connection': [{'from_agent_id': c.from_agent_id,
                          'to_agent_id': c.to_agent_id,
                          'is_approved': c.is_approved}
                         for c in self._connections.values()],
            'groups': [{'group_id': g.group_id,
                       'members': [{'member_agent_id': m.member_agent_id,
                                   'is_admin': m.is_admin} 
                                  for m in g.members]}
                      for g in self._groups.values()],
            'direct_channels': [{'channel_id': dc.channel_id,
                                'members': dc.members}
                               for dc in self._direct_channels.values()],
            'group_channels': [{'channel_id': gc.channel_id,
                               'members': gc.members}
                              for gc in self._group_channels.values()],
            'applications': [{'publisher_agent_id': app.publisher_agent_id,
                             'applicants': app.applicants}
                            for app in self._applications.values()]
        }
    
    def __repr__(self) -> str:
        return f"Environment(id={self._id}, modules={self._modules})"


class Environments:
    """
    Factory that creates all environments from configuration.
    """
    
    def __init__(self, env_config: Dict[str, Any], agents):
        """
        Initialize and create all environments from configuration.
        
        Args:
            env_config: Full environments.json dict
            agents: Agents instance containing all agents
        """
        self._config = env_config
        self._agents = agents
        self._environments: Dict[str, Environment] = {}  # id -> Environment
        self._by_type: Dict[str, List[Environment]] = {}  # type -> List[Environment]
        
        self._create_all_environments()
    
    def _create_all_environments(self):
        """Create all environments based on configuration."""
        for env_type, config in self._config.items():
            quantity = config.get('quantity')
            if quantity is None:
                raise ValueError(f"Environment type '{env_type}' missing required field 'quantity'")
            
            if not isinstance(quantity, int) or quantity < 0:
                raise ValueError(f"Environment type '{env_type}' quantity must be a non-negative integer")
            
            env_list = []
            for i in range(quantity):
                env_id = f"{config['id_prefix']}{i+1:06d}"
                env = Environment(env_id, config, self._agents)
                env_list.append(env)
                self._environments[env_id] = env
            
            self._by_type[env_type] = env_list
    
    def get_by_id(self, env_id: str) -> Optional[Environment]:
        """Get an environment by ID."""
        return self._environments.get(env_id)
    
    def get_by_type(self, env_type: str) -> List[Environment]:
        """Get all environments of a specific type."""
        return self._by_type.get(env_type, [])
    
    def get_all_ids(self) -> List[str]:
        """Get all environment IDs."""
        return list(self._environments.keys())
    
    def get_all_by_type(self) -> Dict[str, List[Environment]]:
        """Get all environments grouped by type."""
        return self._by_type.copy()
    
    @property
    def total_count(self) -> int:
        """Total number of environments across all types."""
        return len(self._environments)
    
    def __getitem__(self, env_id: str) -> Optional[Environment]:
        """Allow indexing by environment ID."""
        return self._environments.get(env_id)
    
    def __contains__(self, env_id: str) -> bool:
        """Check if environment exists by ID."""
        return env_id in self._environments
    
    def __len__(self) -> int:
        return len(self._environments)
    
    def __repr__(self) -> str:
        counts = {t: len(envs) for t, envs in self._by_type.items()}
        return f"Environments(types={counts}, total={self.total_count})"