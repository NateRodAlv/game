import pygame
import sys
import struct
import json
import numpy as np
from dataclasses import dataclass, field

data = bytearray(10000)

with open('memorymap.json', 'r') as f:
    memorymap = json.load(f)

# region function definitions
def readMemory(variable, key):
    return memorymap[variable][key]


def modifyData(variable, newData):
    pointer = int(readMemory(variable, "pointer"))

    # --- BOOLEANs ---
    if isinstance(newData, bool):
        data[pointer] = 1 if newData else 0

    # --- STRs ---
    elif isinstance(newData, str):
        encoded = newData.encode('ascii')
        data[pointer: pointer + len(encoded)] = encoded

    # --- INTs ---
    elif isinstance(newData, int):
        data[pointer: pointer + 4] = newData.to_bytes(4, byteorder='big', signed=True)

    # --- FLOATs ---
    elif isinstance(newData, float):
        data[pointer: pointer + 8] = struct.pack('>d', newData)

    # --- COLORS ---
    elif isinstance(newData, tuple) and len(newData) == 3:
        data[pointer: pointer + 3] = bytes(newData)

    # --- LISTS ---
    elif isinstance(newData, list):
        data[pointer: pointer + len(newData)] = bytes(newData)


def readData(variable):
    epointer = int(readMemory(variable, "epointer"))
    spointer = int(readMemory(variable, "pointer"))
    vtype = readMemory(variable, "type")
    selected = data[spointer:epointer]

    if vtype == "str":
        try:
            return selected.decode('ascii').strip('\x00')
        except UnicodeDecodeError as e:
            return str(e)

    elif vtype == "int":
        if not selected or all(b == 0 for b in selected):
            return 0
        return int.from_bytes(selected, byteorder='big', signed=True)

    elif vtype == "bool":
        return bool(selected[0]) if selected else False

    elif vtype == "tuple":
        return tuple(selected) if len(selected) == 3 else (0, 0, 0)

    elif vtype == "float":
        try:
            return struct.unpack('>d', data[spointer:spointer + 8])[0]
        except Exception:
            return 0.0
    return 0

# endregion

# region tiles stuff
#   tile id  -> the byte value stored in the "loadedTiles" memory region
#   name     -> just for your own reference / debugging
#   source_color -> the RGBA pixel color used in level1.png for this tile
#   render_color -> the color drawn on screen for this tile
#   solid    -> True = blocks movement on ALL sides, like a normal wall.
#               Shorthand for solid_sides = {top, bottom, left, right: True}
#   solid_sides -> (optional) override individual sides instead of using
#               the "solid" shorthand. Any side you omit defaults to
#               whatever "solid" is set to. This is what makes one-way
#               platforms possible:
#                   "solid_sides": {"top": True}
#               means the player lands on it when falling onto the top
#               surface, but can jump up through it from below and walk
#               through it from either side.
#   hazard   -> True = sends the player back to playerSpawn on touch
#   ground   -> True = a touch-sensor: sets onGround without blocking
#               movement (kept for backward compatibility with the
#               original "platform" tile)
#   grid_gap -> True = draw with a 1px gap so tiles look grid-separated
#
# To add a new tile: pick an unused id, add a row below, and (if you
# want it to come from your Aseprite export) give it a unique source_color.
# =====================================================================
EMPTY_TILE_ID = 0

ENTITY_LIBRARY = {
    "slime": {
        "name": "slime",
        "width": 22,
        "height": 18,
        "speed": 0.8,
        "color": (110, 200, 120),
        "max_health": 1,
        "movement": "ground_chaser",
        "gravity": True,
        # Classic stomp enemy: jump on its head to kill it, touch it from
        # the side/below and it kills you instead.
        "on_touch": "stomp_kill",
    },
    "bat": {
        "name": "bat",
        "width": 18,
        "height": 14,
        "speed": 1.5,
        "color": (190, 120, 255),
        "max_health": 1,
        "movement": "flyer",
        "gravity": False,
        # Flies erratically and kills the player on any contact, no
        # stomping it.
        "on_touch": "kill_player",
    },
    "orb": {
        "name": "orb",
        "width": 16,
        "height": 16,
        "speed": 0.6,
        "color": (255, 180, 60),
        "max_health": 1,
        "movement": "floater",
        "gravity": False,
        # Harmless: just disappears when the player touches it.
        "on_touch": "pickup",
    },
}

ENTITY_KIND_IDS = {kind: index for index, kind in enumerate(ENTITY_LIBRARY, start=1)}
ENTITY_RECORD_FORMAT = ">BB2xdddd"
ENTITY_RECORD_SIZE = struct.calcsize(ENTITY_RECORD_FORMAT)
ENEMIES_POINTER = int(memorymap["enemies"]["pointer"])
ENEMIES_EPOINTER = int(memorymap["enemies"]["epointer"])
MAX_ENTITY_SLOTS = (ENEMIES_EPOINTER - ENEMIES_POINTER) // ENTITY_RECORD_SIZE


def _entity_record_pointer(slot):
    if slot < 0 or slot >= MAX_ENTITY_SLOTS:
        raise IndexError(f"Entity slot {slot} is outside the enemies memory region")
    return ENEMIES_POINTER + slot * ENTITY_RECORD_SIZE


def clear_entity_memory():
    data[ENEMIES_POINTER:ENEMIES_EPOINTER] = bytes(ENEMIES_EPOINTER - ENEMIES_POINTER)


def write_entity_memory(entity):
    pointer = _entity_record_pointer(entity.slot)
    data[pointer:pointer + ENTITY_RECORD_SIZE] = struct.pack(
        ENTITY_RECORD_FORMAT,
        1 if entity.alive else 0,
        ENTITY_KIND_IDS[entity.kind],
        entity.x,
        entity.y,
        entity.vx,
        entity.vy,
    )


def read_entity_memory(slot):
    pointer = _entity_record_pointer(slot)
    active, kind_id, x, y, vx, vy = struct.unpack(
        ENTITY_RECORD_FORMAT,
        data[pointer:pointer + ENTITY_RECORD_SIZE],
    )
    return active, kind_id, x, y, vx, vy

TILE_DEFINITIONS = {
    1: {  # Wall
        "name": "wall",
        "source_color": (34, 32, 52, 255),
        "render_color": (100, 100, 100),
        "solid": True,
        "hazard": False,
        "ground": False,
        "grid_gap": True,
    },
    2: {  # Spike / hazard
        "name": "spike",
        "source_color": (255, 0, 0, 255),
        "render_color": (255, 0, 0),
        "solid": False,
        "hazard": True,
        "ground": False,
        "grid_gap": False,
    },
    3: { # One-way platform: land on top, jump through from below/sides
        "name": "one_way_platform",
        "source_color": (0, 255, 0, 255),
        "render_color": (0, 255, 0),
        "solid_sides": {"top": True},
        "hazard": False,
        "ground": False,
        "grid_gap": False,
    },
    4: {
        "name": "slime_spawner",
        "source_color": (0, 0, 255, 255),
        "render_color": (0, 0, 0, 50),
        "solid": False,
        "hazard": False,
        "ground": False,
        "grid_gap": False,
        "spawner": {"entity_type": "slime", "interval": 2.0, "max_active": 2},
    },
    5: {
        "name": "bat_spawner",
        "source_color": (0, 255, 255, 255),
        "render_color": (0, 0, 0, 50),
        "solid": False,
        "hazard": False,
        "ground": False,
        "grid_gap": False,
        "spawner": {"entity_type": "bat", "interval": 3.0, "max_active": 1},
    },
    6: {
        "name": "orb_spawner",
        "source_color": (255, 255, 0, 255),
        "render_color": (0, 0, 0, 50),
        "solid": False,
        "hazard": False,
        "ground": False,
        "grid_gap": False,
        "spawner": {"entity_type": "orb", "interval": 4.0, "max_active": 2},
    },
}

SOURCE_COLOR_TO_TILE_ID = {
    tile_def["source_color"]: tile_id
    for tile_id, tile_def in TILE_DEFINITIONS.items()
}


@dataclass
class Entity:
    slot: int
    kind: str
    spawner_id: int
    x: float
    y: float
    width: int = 20
    height: int = 20
    speed: float = 1.0
    color: tuple = (255, 255, 255)
    max_health: int = 1
    health: int = 1
    alive: bool = True
    vx: float = 0.0
    vy: float = 0.0
    on_ground: bool = False

    def update(self, dt, player_x, player_y, tile_size):
        if not self.alive:
            return

        dx = player_x - self.x
        dy = player_y - self.y
        dist = max(1.0, (dx * dx + dy * dy) ** 0.5)

        behavior = ENTITY_LIBRARY[self.kind]
        movement = behavior.get("movement", "ground_chaser")

        if movement == "ground_chaser":
            self.vx = (dx / dist) * self.speed
        elif movement == "flyer":
            self.vx = (dx / dist) * self.speed
            self.vy = (dy / dist) * self.speed * 0.75
        elif movement == "floater":
            self.vx = (dx / dist) * self.speed * 0.5
            self.vy = (dy / dist) * self.speed * 0.5

        if behavior.get("gravity", False):
            self.vy += readData("gravity") * dt * 60

        next_x = self.x + self.vx * dt * 60
        horizontal_side = "left" if self.vx > 0 else "right"
        horizontal_points = [
            (next_x, self.y),
            (next_x, self.y + self.height - 1),
        ] if self.vx > 0 else [
            (next_x + self.width - 1, self.y),
            (next_x + self.width - 1, self.y + self.height - 1),
        ]
        horizontal_collision = any(
            flags_at_pixel(x, y)["solid_sides"][horizontal_side]
            for x, y in horizontal_points
        )
        if horizontal_collision:
            moving_right = self.vx > 0
            self.vx = 0.0
            if moving_right:
                self.x = (int(next_x) // tile_size) * tile_size - self.width
            else:
                self.x = (int(next_x + self.width - 1) // tile_size + 1) * tile_size
        else:
            self.x = next_x

        next_y = self.y + self.vy * dt * 60
        self.on_ground = False
        if self.vy >= 0:
            vertical_side = "top"
            vertical_points = [
                (self.x, next_y + self.height - 1),
                (self.x + self.width - 1, next_y + self.height - 1),
            ]
        else:
            vertical_side = "bottom"
            vertical_points = [
                (self.x, next_y),
                (self.x + self.width - 1, next_y),
            ]

        blocked = False
        collision_tile_edge = None
        for x, y in vertical_points:
            flags = flags_at_pixel(x, y)
            if not flags["solid_sides"][vertical_side]:
                continue

            tile_top = (int(y) // tile_size) * tile_size
            if self.vy > 0:
                previous_bottom = self.y + self.height
                if previous_bottom <= tile_top + 1:
                    blocked = True
                    collision_tile_edge = tile_top
            elif self.vy < 0:
                tile_bottom = tile_top + tile_size
                previous_top = self.y
                if previous_top >= tile_bottom - 1:
                    blocked = True
                    collision_tile_edge = tile_bottom

        if blocked:
            if self.vy > 0:
                self.on_ground = True
                self.y = collision_tile_edge - self.height
            elif self.vy < 0:
                self.y = collision_tile_edge
            self.vy = 0.0
        else:
            self.y = next_y

        write_entity_memory(self)

    def draw(self, screen, camera_x):
        if not self.alive:
            return
        pygame.draw.rect(screen, self.color, (self.x - camera_x, self.y, self.width, self.height))


# region entity touch scripts
#
# Every entity kind in ENTITY_LIBRARY has an "on_touch" key naming one of
# the functions below (see TOUCH_HANDLERS). A handler is called once per
# frame per entity the player is overlapping, and receives:
#   entity -> the Entity instance being touched
#   stomp  -> True if the player is falling and landed on the entity's
#             top half (a classic Mario-style stomp), False otherwise
#
# A handler must return one of these outcome strings:
#   "none"        -> nothing happens to the player
#   "kill_entity" -> the entity dies, player is unharmed
#   "kill_player" -> the player dies (resets to spawn, all entities reset)
#   "bounce"      -> the entity dies AND the player gets a small upward
#                     bounce (useful for stomp-kill enemies)
#
# To add a new behavior: write a function with this same signature below,
# register it in TOUCH_HANDLERS under a new name, then reference that name
# from any entity's "on_touch" field in ENTITY_LIBRARY.

def touch_stomp_kill(entity, stomp):
    """Landing on top kills the entity and bounces the player; touching
    it any other way kills the player."""
    if stomp:
        return "bounce"
    return "kill_player"


def touch_kill_player(entity, stomp):
    """Always lethal to the player, no matter how it's touched."""
    return "kill_player"


def touch_pickup(entity, stomp):
    """Harmless - just despawns when touched."""
    return "kill_entity"


TOUCH_HANDLERS = {
    "stomp_kill": touch_stomp_kill,
    "kill_player": touch_kill_player,
    "pickup": touch_pickup,
}


def resolve_entity_collisions(pre_collision_y_acc):
    """Checks the player's rect against every living entity and runs its
    on_touch script. pre_collision_y_acc is the player's vertical speed
    from before this frame's tile collision resolved it, used to tell a
    downward stomp apart from a sideways/upward touch. Returns True if
    the player died this frame."""
    player_w = int(readData("playerWidth"))
    player_h = int(readData("playerHeight"))
    player_rect = pygame.Rect(
        readData("playerX"), readData("playerY"), player_w, player_h
    )

    for entity in entities:
        if not entity.alive:
            continue

        entity_rect = pygame.Rect(entity.x, entity.y, entity.width, entity.height)
        if not player_rect.colliderect(entity_rect):
            continue

        handler_name = ENTITY_LIBRARY[entity.kind].get("on_touch")
        handler = TOUCH_HANDLERS.get(handler_name)
        if handler is None:
            continue

        stomp = (
            pre_collision_y_acc > 0
            and (player_rect.bottom - pre_collision_y_acc) <= entity_rect.centery
        )

        outcome = handler(entity, stomp)

        if outcome in ("kill_entity", "bounce"):
            entity.alive = False
            write_entity_memory(entity)

        if outcome == "bounce":
            modifyData("playerYAcc", jump_velocity_for_bounce())

        if outcome == "kill_player":
            kill_player()
            return True

    return False


def jump_velocity_for_bounce():
    """A stomp bounce reuses a fraction of the normal jump strength."""
    return float(readData("jumpVelocity")) * 0.6


def kill_player():
    if readData("iframes") < 0:
        spawn = readData("playerSpawn")
        modifyData("playerX", float(spawn[0]))
        modifyData("playerY", float(spawn[1]))
        modifyData("playerXAcc", 0.0)
        modifyData("playerYAcc", 0.0)
        modifyData("onGround", False)
        modifyData("fastFall", True)
        reset_entities()


def reset_entities():
    """Clears every active entity and lets spawners start fresh."""
    entities.clear()
    clear_entity_memory()
    for spawner in spawners:
        spawner.cooldown = 0.0
# endregion


@dataclass
class Spawner:
    spawner_id: int
    x: float
    y: float
    entity_type: str
    interval: float = 2.0
    max_active: int = 2
    cooldown: float = 0.0
    width: int = 32
    height: int = 32

    def update(self, dt, entity_list):
        self.cooldown = max(0.0, self.cooldown - dt)
        active_count = sum(
            1 for entity in entity_list
            if entity.spawner_id == self.spawner_id and entity.alive
        )
        if self.cooldown <= 0 and active_count < self.max_active:
            self.spawn(entity_list)

    def spawn(self, entity_list):
        if self.entity_type not in ENTITY_LIBRARY:
            raise ValueError(f"Unknown entity type: {self.entity_type}")

        defn = ENTITY_LIBRARY[self.entity_type]
        occupied_slots = {entity.slot for entity in entity_list if entity.alive}
        free_slot = next(
            (slot for slot in range(MAX_ENTITY_SLOTS) if slot not in occupied_slots),
            None,
        )
        if free_slot is None:
            return

        entity_list.append(
            Entity(
                slot=free_slot,
                kind=self.entity_type,
                spawner_id=self.spawner_id,
                x=self.x + self.width * 0.5,
                y=self.y + self.height * 0.5,
                width=defn["width"],
                height=defn["height"],
                speed=defn["speed"],
                color=defn["color"],
                max_health=defn["max_health"],
                health=defn["max_health"],
            )
        )
        write_entity_memory(entity_list[-1])
        self.cooldown = self.interval


def build_spawners_from_tiles(tile_layout, cols, rows):
    spawners = []
    for index, tile_id in enumerate(tile_layout):
        tile_def = TILE_DEFINITIONS.get(tile_id)
        if tile_def is None or "spawner" not in tile_def:
            continue
        col = index % cols
        row = index // cols
        config = tile_def["spawner"]
        spawners.append(
            Spawner(
                spawner_id=len(spawners),
                x=col * TILE_SIZE,
                y=row * TILE_SIZE,
                entity_type=config["entity_type"],
                interval=config.get("interval", 2.0),
                max_active=config.get("max_active", 1),
                width=40,
                height=40,
            )
        )
    return spawners


def _resolve_solid_sides(tile_def):
    base = tile_def.get("solid", False)
    sides = {"top": base, "bottom": base, "left": base, "right": base}
    sides.update(tile_def.get("solid_sides", {}))
    return sides


for _tile_def in TILE_DEFINITIONS.values():
    _tile_def["solid_sides"] = _resolve_solid_sides(_tile_def)

_EMPTY_TILE_FLAGS = {
    "solid_sides": {"top": False, "bottom": False, "left": False, "right": False},
    "hazard": False,
    "ground": False,
}


def get_tile_def(tile_id):
    """Look up a tile's properties, falling back to 'empty' for unknown ids."""
    return TILE_DEFINITIONS.get(tile_id, _EMPTY_TILE_FLAGS)
#endregion


# --- Collision Checker ---
def check_tile_at_pixel(pixel_x, pixel_y):
    """Calculates the specific tile index in VRAM based on screen coordinates."""
    tile_size = int(readData("tileSize"))
    cols = int(readData("mapCols"))
    rows = int(readData("mapRows"))

    tile_col = int(pixel_x // tile_size)
    tile_row = int(pixel_y // tile_size)

    # Boundary constraint safety fallback
    if tile_col < 0 or tile_col >= cols or tile_row < 0 or tile_row >= rows:
        return 1  

    # Translate 2D map column/row to a 1D sequence offset
    array_index = (tile_row * cols) + tile_col

    # Look directly inside the memory array
    map_start = int(readMemory("loadedTiles", "pointer"))
    return data[map_start + array_index]


def flags_at_pixel(pixel_x, pixel_y):
    return get_tile_def(check_tile_at_pixel(pixel_x, pixel_y))


# region Level Initialization 
MAP_COLS = 100
MAP_ROWS = 15
TILE_SIZE = 40

try:
    map_image = pygame.image.load('level1.png')

    level_layout = []
    for y in range(MAP_ROWS):
        for x in range(MAP_COLS):
            color = map_image.get_at((x, y))
            level_layout.append(SOURCE_COLOR_TO_TILE_ID.get(tuple(color), EMPTY_TILE_ID))

except (pygame.error, FileNotFoundError):
    print("Could not find level1.png! Falling back to flat map.")
    # Fallback backup map if file is missing
    border_row = [1] * MAP_COLS
    empty_row = [1] + [0] * (MAP_COLS - 2) + [1]
    level_layout = []
    level_layout.extend(border_row)
    for _ in range(MAP_ROWS - 2):
        level_layout.extend(empty_row)
    level_layout.extend(border_row)


pygame.init()

modifyData("playerWidth", 25)
modifyData("playerHeight", 35)
modifyData("tileSize", TILE_SIZE)
modifyData("mapCols", MAP_COLS)
modifyData("mapRows", MAP_ROWS)
modifyData("screenWidth", 800)
modifyData("screenHeight", 600)

modifyData("gravity", 0.3)
modifyData("jumpVelocity", -8.5)
modifyData("maxXAcc", 5.0)
modifyData("xAccStep", 0.4)
modifyData("frictionDivisor", 10.0)
modifyData("cameraLookahead", 400.0)
modifyData(
    "cameraMaxX",
    float(max(0, MAP_COLS * TILE_SIZE - 800)),
)

# Player surface sized from the memory-mapped width/height
my_surface = pygame.Surface((int(readData("playerWidth")), int(readData("playerHeight"))))
my_surface.fill((255, 0, 0))

screen = pygame.display.set_mode((int(readData("screenWidth")), int(readData("screenHeight"))))

modifyData("loadedTiles", level_layout)

# Standard Player Spawn
modifyData("playerSpawn", (100, 100, 0))
modifyData("playerX", float(readData("playerSpawn")[0]))
modifyData("playerY", float(readData("playerSpawn")[1]))
modifyData("playerXAcc", 0.0)
modifyData("playerYAcc", 0.0)
modifyData("onGround", False)
modifyData("cameraX", 0.0)

clear_entity_memory()
entities = []
spawners = build_spawners_from_tiles(level_layout, MAP_COLS, MAP_ROWS)
#endregion

#region game loop
running = True
clock = pygame.time.Clock()
while running:
    screen.fill((0, 0, 0))

    # Pull this frame's tunables out of the data array once, up front,
    # so anything editing memory mid-game (or a future debug overlay)
    # is picked up consistently for the whole frame.
    tile_size = int(readData("tileSize"))
    player_w = int(readData("playerWidth"))
    player_h = int(readData("playerHeight"))
    gravity = readData("gravity")
    jump_velocity = readData("jumpVelocity")
    max_x_acc = readData("maxXAcc")
    x_acc_step = readData("xAccStep")
    friction_divisor = readData("frictionDivisor")
    camera_lookahead = readData("cameraLookahead")
    camera_max_x = readData("cameraMaxX")

    # --- Camera Tracking Core Logic ---
    px = readData("playerX")
    target_camera_x = px - camera_lookahead

    # Keep camera clamped within bounds
    if target_camera_x < 0.0:
        target_camera_x = 0.0
    if target_camera_x > camera_max_x:
        target_camera_x = camera_max_x
    modifyData("cameraX", target_camera_x)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                if readData("dashTimer") < 1:
                    modifyData("dashTimer", 90)
                    modifyData("iframes", 20)
                    modifyData("playerXAcc", np.sign(readData("playerXAcc")) * 25)

    keys = pygame.key.get_pressed()

    # --- Horizontal Acceleration Input ---
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        if readData("playerXAcc") >= -max_x_acc:
            modifyData("playerXAcc", readData("playerXAcc") - x_acc_step)
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        if readData("playerXAcc") <= max_x_acc:
            modifyData("playerXAcc", readData("playerXAcc") + x_acc_step)
    else:
        modifyData("playerXAcc", readData("playerXAcc") - (readData("playerXAcc") / friction_divisor))

    modifyData("dashTimer", readData("dashTimer") - 1)
    # --- Vertical Acceleration Input (Jumping & Gravity) ---
    modifyData("playerYAcc", readData("playerYAcc") + gravity)

    if (keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_SPACE]) and readData("onGround"):
        modifyData("playerYAcc", jump_velocity)
        modifyData("onGround", False)
        modifyData("fastFall", True)
        
    if (keys[pygame.K_DOWN] or keys[pygame.K_s] and not readData("onGround")):
        if readData("fastFall"):
            modifyData("playerYAcc", 4.0)
            modifyData("fastFall", False)
            print(readData("fastFall"))
            
    if readData("onGround"):
        modifyData("dashTimer", readData("dashTimer") - 5)
    
    modifyData("iframes", readData("iframes") - 1)

    for spawner in spawners:
        spawner.update(1 / 60, entities)

    for entity in entities:
        entity.update(
            1 / 60,
            readData("playerX"),
            readData("playerY"),
            tile_size,
        )

    for entity in entities:
        if not entity.alive:
            write_entity_memory(entity)
    entities[:] = [entity for entity in entities if entity.alive]

    # --- PHYSICS ENGINE SEPARATION ---
    # 1. Update and process X axis collision movements
    x_acc = readData("playerXAcc")
    new_px = readData("playerX") + x_acc
    py = readData("playerY")

    x_corners = [
        (new_px, py),
        (new_px + player_w, py),
        (new_px, py + player_h - 1),
        (new_px + player_w, py + player_h - 1),
    ]
    x_flags = [flags_at_pixel(cx, cy) for cx, cy in x_corners]

    if x_acc > 0.0:
        x_side = "left"
    elif x_acc < 0.0:
        x_side = "right"
    else:
        x_side = None

    if x_side and any(f["solid_sides"][x_side] for f in x_flags):
        modifyData("playerXAcc", 0.0)
    elif any(f["hazard"] for f in x_flags):
        kill_player()
    elif any(f["ground"] for f in x_flags):
        modifyData("onGround", True)
    else:
        modifyData("playerX", new_px)

    # 2. Update and process Y axis collision movements
    px = readData("playerX")
    y_acc = readData("playerYAcc")
    new_py = py + y_acc
    moving_down = y_acc > 0.0


    if moving_down:
        y_side = "top"
        y_points = [(px, new_py + player_h - 1), (px + player_w - 1, new_py + player_h - 1)]
        prev_edge = py + player_h
    else:
        y_side = "bottom"
        y_points = [(px, new_py), (px + player_w - 1, new_py)]
        prev_edge = py

    blocked = False
    for cx, cy in y_points:
        flags = flags_at_pixel(cx, cy)
        if not flags["solid_sides"][y_side]:
            continue
        tile_top = (int(cy) // tile_size) * tile_size
        if moving_down:
            if prev_edge <= tile_top + 1:
                blocked = True
        else:
            tile_bottom = tile_top + tile_size
            if prev_edge >= tile_bottom - 1:
                blocked = True

    modifyData("onGround", False)
    if blocked:
        if moving_down:
            modifyData("onGround", True)
        modifyData("playerYAcc", 0.0)
    else:
        modifyData("playerY", new_py)

    # --- Entity touch scripts (stomp/hurt/pickup/etc.) ---
    # y_acc is the player's fall speed from before tile collision zeroed
    # it, which is what tells a downward stomp apart from any other kind
    # of touch.
    resolve_entity_collisions(y_acc)

    # --- Rendering Engine ---
    cam_x = readData("cameraX")

    spointer = int(readMemory("loadedTiles", "pointer"))
    epointer = int(readMemory("loadedTiles", "epointer"))
    current_map_bytes = data[spointer:epointer]

    cols = int(readData("mapCols"))
    screen_width = int(readData("screenWidth"))

    for index, tile_id in enumerate(current_map_bytes):
        if tile_id == EMPTY_TILE_ID:
            continue

        tile_def = TILE_DEFINITIONS.get(tile_id)
        if tile_def is None:
            continue 

        col = index % cols
        row = index // cols

        world_x = col * tile_size
        world_y = row * tile_size

        screen_x = world_x - cam_x
        screen_y = world_y

        if -tile_size <= screen_x <= screen_width:
            size = tile_size - 1 if tile_def["grid_gap"] else tile_size
            pygame.draw.rect(screen, tile_def["render_color"], (screen_x, screen_y, size, size))

    final_x = readData("playerX")
    final_y = readData("playerY")
    screen.blit(my_surface, (final_x - cam_x, final_y))

    for spawner in spawners:
        pygame.draw.rect(screen, (60, 60, 60), (spawner.x - cam_x, spawner.y, spawner.width, spawner.height), 1)
    for entity in entities:
        entity.draw(screen, cam_x)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
#endregion