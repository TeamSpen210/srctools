"""Parses Portal 2's Puzzlemaker puzzles."""
from srctools.property_parser import Property
from srctools.vec import Vec, Vec_tuple
import srctools

from enum import Enum
from datetime import datetime as DateTime

from typing import Union, List, Dict, Any


class ConnVis(Enum):
    """Type of connection visibility."""
    ANTLINE = 0
    SIGNAGE = 1
    NONE    = 2


class ConnType(Enum):
    """Types of item connections.
    
    The values are the types in the file, without 'CONNECTION_'.
    """
    # Normal I/O
    STANDARD = 'STANDARD'
    # Fizzler -> models, reused for coop/normal doors.
    FIZZLER = 'BARRIER_ANCHOR_TO_EXTENT'
    # Cube dropper -> cube
    CUBE_DROPPER = 'BOX_DROPPER'
    # Gel dropper -> Gel
    GEL_DROPPER = 'PAINT_DROPPER'
    # Faith plate -> target
    FAITH_PLATE = 'CATAPULT_TARGET'
 
class PropType(Enum):
    """The types of data in item properties."""
    # Special
    SUBTYPE = 'subtype'  # Buttontype, etc - int.
    CONN_COUNT = 'conncount'  # Connection count for normal and tbeam.
    PANEL_ANGLE = 'panel_angle'  # ramp_xx_deg_open
    
    # Generic
    INT = 'int'
    FLOAT = 'float'
    BOOL = 'bool'
    STR = 'str'
    VEC = 'vec'


class ItemProps(Enum):
    """Properties usable for items."""
    CONN_COUNT = 'CONNECTION_COUNT'
    CONN_COUNT_TBEAM = 'CONNECTION_COUNT_POLARITY'
    
    # Subtypes:
    CUBE_TYPE = 'CUBE_TYPE'
    GEL_TYPE = 'PAINT_TYPE'
    GLASS_TYPE = 'BARRIER_TYPE'
    FIZZ_TYPE = 'BARRIER_HAZARD_TYPE'
    BUTTON_TYPE = 'BUTTON_TYPE'
    
    # Removes dropper
    CUBE_HAS_DROPPER = 'DROPPER_ENABLED'
    # Inverted on dropper item instance, otherwise normal.
    CUBE_AUTO_DROP = 'AUTO_DROP_CUBE'
    CUBE_AUTO_RESPAWN = 'AUTO_RESPAWN_CUBE'
    # Never set to true, unknown purpose.
    CUBE_FALL_DOWN = 'DROPPER_FALL_STRAIGHT_DOWN'
    
    TIMER_DELAY = 'TIMER_DELAY'
    TIMER_SOUND = 'TIMER_SOUND'
    
    PISTON_TOP = 'PISTON_LIFT_TOP_LEVEL'
    PISTON_BOTTOM = 'PISTON_LIFT_BOTTOM_LEVEL'
    PISTON_START_UP = 'PISTON_LIFT_START_UP'
    
    FAITH_SPEED = 'CATAPULT_SPEED'
    FAITH_TARGETNAME = 'TARGET_NAME'
    FAITH_VERT_ALIGN = 'VERTICAL_ALIGNMENT'
    
    # Is angled or flip panel portalable?
    PANEL_PORTALABLE = 'PORTALABLE'
    # Ramp degrees, in ramp_xx_deg_open format
    PANEL_ANGLE = 'ANGLED_PANEL_ANIMATION'
    # Seems to specify glass/angled type?
    PANEL_TYPE = 'ANGLED_PANEL_TYPE'
    
    GEL_STREAK = 'ALLOW_STREAK_PAINT'
    GEL_FLOW = 'PAINT_FLOW_TYPE'
    GEL_EXPORT_TYPE = 'PAINT_EXPORT_TYPE'
    
    TRACK_OSCILLATE = 'RAIL_OSCILLATE'
    TRACK_START_FRAC = 'RAIL_STARTING_POSITION'
    TRACK_SPEED = 'RAIL_SPEED'
    TRACK_DIRECTION = 'RAIL_TRAVEL_DIRECTION'
    TRACK_DISTANCE = 'RAIL_TRAVEL_DISTANCE'
    
    CORR_IS_COOP = 'DOOR_IS_COOP'
    
    # Bool 'Start x' checkboxes
    ST_DEPLOYED = 'START_DEPLOYED'
    ST_ENABLED = 'START_ENABLED'
    ST_REVERSED = 'START_REVERSED'
    ST_OPEN = 'START_OPEN'
    ST_LOCKED = 'COOP_EXIT_STARTS_LOCKED'
    ST_ACTIVE = 'RAIL_START_ACTIVE'  # Disabled if non-oscillating
    

# Property -> type of prop
ITEM_PROPS = {
    # Hidden things.
    ItemProps.CONN_COUNT: PropType.CONN_COUNT,
    ItemProps.CONN_COUNT_TBEAM: PropType.CONN_COUNT,
    
    ItemProps.CORR_IS_COOP: PropType.BOOL,
    ItemProps.TIMER_DELAY: PropType.INT,
    ItemProps.TIMER_SOUND: PropType.BOOL,
    
    ItemProps.PANEL_ANGLE: PropType.PANEL_ANGLE,
    ItemProps.PANEL_PORTALABLE: PropType.BOOL,
    ItemProps.PANEL_TYPE: PropType.INT,
    
    ItemProps.GEL_FLOW: PropType.INT,
    ItemProps.GEL_EXPORT_TYPE: PropType.INT,
    ItemProps.GEL_STREAK: PropType.BOOL,
    
    ItemProps.CUBE_AUTO_DROP: PropType.BOOL,
    ItemProps.CUBE_AUTO_RESPAWN: PropType.BOOL,
    ItemProps.CUBE_HAS_DROPPER: PropType.BOOL,
    ItemProps.CUBE_FALL_DOWN: PropType.BOOL,
    
    ItemProps.FAITH_SPEED: PropType.FLOAT,
    ItemProps.FAITH_TARGETNAME: PropType.STR,
    ItemProps.FAITH_VERT_ALIGN: PropType.BOOL,
    
    ItemProps.TRACK_DIRECTION: PropType.VEC,
    ItemProps.TRACK_OSCILLATE: PropType.BOOL,
    ItemProps.TRACK_SPEED: PropType.FLOAT,
    ItemProps.TRACK_START_FRAC: PropType.FLOAT,
    ItemProps.TRACK_DISTANCE: PropType.FLOAT,
    
    ItemProps.PISTON_BOTTOM: PropType.INT,
    ItemProps.PISTON_TOP: PropType.INT,
    ItemProps.PISTON_START_UP: PropType.BOOL,
}

# Check they all have a type.
prop = None
for prop in ItemProps:
    # Start open, enabled, etc are all bool.
    if prop.name[:3] == 'ST_':
        ITEM_PROPS[prop] = PropType.BOOL
    if prop.name[-4:] == 'TYPE':
        ITEM_PROPS.setdefault(prop, PropType.SUBTYPE)
    
    assert prop in ITEM_PROPS, prop
del prop


def hex_to_date(hex_time: str) -> DateTime:
    """Convert from the hex format in P2Cs to a DateTime."""
    try:
        time = int(hex_time, base=16)
    except ValueError:
        return DateTime.now()
    else:
        return DateTime.fromtimestamp(time)


def date_to_hex(time: DateTime) -> str:
    """Convert from the hex format in P2Cs to a DateTime."""
    return hex((time - DateTime(1970, 1, 1, 10, 0)).total_seconds())


class Item:
    """Represents an item."""
    def __init__(
        self,
        id: int,
        type: str,
        pos: Vec,
        local_pos: Vec,
        angles: Vec,
        facing: Vec,
        conn_vis: ConnVis,
        props: Dict[ItemProps, Any],
        ):
        self.id = id
        self.type = type
        self.pos = pos
        self.offset = local_pos
        self.angles = angles
        self.facing = facing
        self.conn_vis = conn_vis
        self.props = props
        
    def __repr__(self):
        return '<"{}" Item at ({})>'.format(self.type, self.pos)
        
    @classmethod
    def parse(cls, keyvalues: Property):
        """Parse an item from property data."""
        item_id = keyvalues.int('index')
        item_type = keyvalues['Type']
        is_deleteable = keyvalues.bool('Deletable')
        loc = keyvalues.vec('VoxelPos')
        local_off = keyvalues.vec('LocalPos')
        angles = keyvalues.vec('Angles')
        facing = keyvalues.vec('Facing')
        conn_vis = ConnVis(keyvalues.int('ConnectionVisibility'))
        
        props = {}
        
        for kval in keyvalues:
            name = kval.real_name.upper()
            if name[:14] != 'ITEM_PROPERTY_':
                continue
            prop_name = ItemProps(name[14:])
            value = kval.value
            prop_type = ITEM_PROPS[prop_name]
            if prop_type is PropType.INT:
                value = srctools.conv_int(value)
            elif prop_type is PropType.BOOL:
                value = srctools.conv_bool(value)
            elif prop_type is PropType.FLOAT:
                value = srctools.conv_float(value)
            elif prop_type is PropType.STR:
                pass
            elif prop_type is PropType.VEC:
                value = Vec.from_str(value)
            
            elif prop_type is PropType.PANEL_ANGLE:
                # ramp_xx_deg_open
                value = srctools.conv_int(value[5:7])
            elif prop_type is PropType.SUBTYPE:
                value = srctools.conv_int(value)
            elif prop_type is PropType.CONN_COUNT:
                value = srctools.conv_int(value)
            else:
                raise ValueError('Unknown prop type {}!'.format(prop_type))
            
            props[prop_name] = value
        
        return cls(
            item_id,
            item_type,
            loc,
            local_off,
            angles,
            facing,
            conn_vis,
            props,
        )


class Connection:
    """Represents a connection between two items.
    
    The type can be one of thw standard ConnTypes, or a string for custom connections.
    """
    __slots__ = ['sender', 'receiver', 'type']
    
    def __init__(
        self,
        sender: Item,
        receiver: Item,
        type: Union[ConnType, str],
    ):
        self.sender = sender
        self.receiver = receiver
        self.type = type
        
    def __repr__(self):
        if isinstance(self.type, ConnType):
            return '<{} Connection from {!r} to {!r}>'.format(
                self.type.name, 
                self.sender.id, 
                self.receiver.id,
            )
        return '<Custom "{}" Connection from {!r} to {!r}>'.format(
                self.type, 
                self.sender.id, 
                self.receiver.id,
            )


class Puzzle:
    """Represents a P2C file."""
    def __init__(
        self,
        coop: bool,
        size: Vec,
        name: str,
        desc: str,
        items: Dict[int, Item],
        blocks: Dict[Vec_tuple, 'Block'],
        conn: List['Connection'],
    ):
        self.is_coop = coop
        self.size = size
        self.name = name
        self.desc = desc
        
        self.blocks = blocks
        self.items = items
        self.conn = conn
        
    @classmethod
    def parse(cls, props: Property):
        """Given the keyvalues, parse a puzzle out."""
        if props.name is None:
            # Root key name.
            props = props.find_key('portal2_puzzle')
        is_coop = props.bool('Coop')
        name = props['Title', '']
        desc = props['Description', '']
        map_size = props.vec('ChamberSize')
        
        items = {}
        blocks = {}
        conn = []
        
        for item_prop in props.find_all('Items', 'Item'):
            item = Item.parse(item_prop)
            items[item.id] = item
            
        for conn_prop in props.find_all('Connections', 'Connection'):
            inp_item = items[conn_prop.int('receiver')]
            out_item = items[conn_prop.int('sender')]
            conn_type = conn_prop['Type']
            try:
                # Strip 'CONNECTION_'
                conn_type = ConnType(conn_type[11:].upper())
            except ValueError:
                # Not a standard one, a custom one?
                pass
            
            conn.append(Connection(
                inp_item,
                out_item,
                conn_type,
            ))
        
        return cls(
            is_coop,
            map_size,
            name, 
            desc,
            items,
            blocks, 
            conn,
        )
