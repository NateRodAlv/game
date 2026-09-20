import pygame
import sys
import asyncio
import os
import struct
import json
import numpy as np
from dataclasses import dataclass, field

# =====================================================================
# FRAGILE ZONES — read this before you go hunting for glitches
# =====================================================================
# Design rule: the game must be fully beatable start-to-finish with zero
# glitches. Every fragile spot below is optional, discoverable, and never
# required. Each new ability you add should get its own entry here.
#
#   - dashTimer / invincibilityTimer: no clamping on how they stack.
#     Dash-canceling into things fast enough can push these outside their
#     "intended" range.
#   - playerHealth: not clamped on the way down either - a hit that
#     overkills exact-0 still triggers death correctly (see
#     damage_player), but the raw value can transiently go negative
#     before kill_player() resets it. Harmless, but if you ever read
#     health for a UI bar without clamping for display, that's on you.
#   - Entity record struct (ENTITY_RECORD_FORMAT): state/speed_mult/
#     target_offset are raw bytes with no validation on read. A state
#     byte that doesn't match a known constant just falls through to a
#     default behavior right now (see MOVEMENT_HANDLERS.get fallback) —
#     but if you stop guarding that fallback later, a corrupted state
#     byte could point at real, unintended behavior.
#   - Room loading (load_room): the shared tile region is NOT fully
#     zeroed before a new room's tiles are written in, only overwritten
#     tile-by-tile up to the new room's size. A room smaller than the
#     previous one will leave old tiles behind past its own bounds.
#     That's intentional — don't "fix" it.
#   - check_tile_at_pixel: there is NO fallback wall at a room's edges
#     anymore - walking past the last row/column of tiles is just open
#     space, not a hidden boundary. This means an unbordered room lets
#     you walk (or dash) straight out of the intended play area. That's
#     no longer a corruption lever, it's a normal reachable mechanic -
#     if you want a room to actually be bounded, paint real wall tiles
#     around its edges in Tiled so collision matches what's rendered.
#     Position values themselves are still just floats with no clamping,
#     so wandering far enough negative/positive is still fair game if
#     you want a genuine "walked off the edge of the world" glitch.
#   - render_tiles_live (toggled per-room via the liveMemoryRendering map
#     property): deliberately draws tiles by raw offset into the whole
#     shared `data` buffer rather than the room's own bounded slice, so
#     wandering past a room's edges can visibly show other memory
#     reinterpreted as tiles. This one's not a hidden lever at all - it's
#     an intentional, designer-opted-into rendering mode. Off by default
#     per room; only rooms you explicitly flag get it.
# =====================================================================

data = bytearray(10000)

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # pygbag (and some other embedded/exec-based runners) don't always
    # define __file__ for the entry script. Fall back to the current
    # working directory - under pygbag this is already the root of the
    # extracted game archive, so plain relative paths still resolve
    # correctly.
    BASE_DIR = os.getcwd()


def resolve_path(path):
    """Turns a path from JSON (room files, tileset images, etc.) into one
    anchored to this script's folder, so it works no matter what
    directory the game is launched from. Absolute paths pass through
    unchanged."""
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


with open(resolve_path('memorymap.json'), 'r') as f:
    memorymap = json.load(f)

# region function definitions
def readMemory(variable, key):
    return memorymap[variable][key]


def modifyData(variable, newData):
    pointer = int(readMemory(variable, "pointer"))

    if isinstance(newData, bool):
        data[pointer] = 1 if newData else 0

    elif isinstance(newData, str):
        encoded = newData.encode('ascii')
        data[pointer: pointer + len(encoded)] = encoded

    elif isinstance(newData, int):
        data[pointer: pointer + 4] = newData.to_bytes(4, byteorder='big', signed=True)

    elif isinstance(newData, float):
        data[pointer: pointer + 8] = struct.pack('>d', newData)

    elif isinstance(newData, tuple) and len(newData) == 3:
        data[pointer: pointer + 3] = bytes(newData)

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


# region abilities
# Simple bitfield in the "unlockedAbilities" memory int. Framework only —
# wire pickups/rooms to call unlock_ability() when you design progression.
ABILITY_BITS = {
    "dash": 1 << 0,
    "wallJump": 1 << 1,
    "doubleJump": 1 << 2,
    "ledgeGrab": 1 << 3,
}


def has_ability(name):
    return bool(readData("unlockedAbilities") & ABILITY_BITS[name])


def unlock_ability(name):
    modifyData("unlockedAbilities", readData("unlockedAbilities") | ABILITY_BITS[name])
# endregion


# region tiles
# Tile "id" here should match the custom integer tile property you set on
# each tile in your Tiled tileset (see load_room for the exact expected
# JSON shape). render_color is used as a fallback whenever no tileset
# image is loaded (or a GID has no matching tile image).
EMPTY_TILE_ID = 0

TILE_DEFINITIONS = {
    1: {
        "name": "wall",
        "render_color": (100, 100, 100),
        "solid": True,
        "hazard": False,
        "ground": False,
        "grid_gap": True,
    },
    2: {
        "name": "spike",
        "render_color": (255, 0, 0),
        "solid": False,
        "hazard": True,
        "damage": 1,
        "ground": False,
        "grid_gap": False,
    },
    3: {
        "name": "one_way_platform",
        "render_color": (0, 255, 0),
        "solid_sides": {"top": True},
        "hazard": False,
        "ground": False,
        "grid_gap": False,
        "one_way": True,
    },
    4: {
        "name": "slime_spawner",
        "render_color": (0, 0, 0),
        "solid": False,
        "hazard": False,
        "ground": False,
        "grid_gap": False,
        "spawner": {"entity_type": "slime", "interval": 2.0, "max_active": 2},
    },
    5: {
        "name": "bat_spawner",
        "render_color": (0, 0, 0),
        "solid": False,
        "hazard": False,
        "ground": False,
        "grid_gap": False,
        "spawner": {"entity_type": "bat", "interval": 3.0, "max_active": 1},
    },
    6: {
        "name": "orb_spawner",
        "render_color": (0, 0, 0),
        "solid": False,
        "hazard": False,
        "ground": False,
        "grid_gap": False,
        "spawner": {"entity_type": "orb", "interval": 4.0, "max_active": 2},
    },
}


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
    return TILE_DEFINITIONS.get(tile_id, _EMPTY_TILE_FLAGS)


def check_tile_at_pixel(pixel_x, pixel_y):
    tile_size = int(readData("tileSize"))
    cols = int(readData("mapCols"))
    rows = int(readData("mapRows"))

    tile_col = int(pixel_x // tile_size)
    tile_row = int(pixel_y // tile_size)

    if tile_col < 0 or tile_col >= cols or tile_row < 0 or tile_row >= rows:
        # No invisible boundary here anymore - out of a room's tile grid
        # is just empty, non-solid space, same as any other empty tile.
        # Collision now matches exactly what's rendered, so if you want a
        # room to actually be bounded, paint real wall tiles around its
        # edges in Tiled. See FRAGILE ZONES at the top of this file for
        # what this trades away.
        return EMPTY_TILE_ID

    array_index = (tile_row * cols) + tile_col
    map_start = int(readMemory("loadedTiles", "pointer"))
    return data[map_start + array_index]


def flags_at_pixel(pixel_x, pixel_y):
    return get_tile_def(check_tile_at_pixel(pixel_x, pixel_y))
# endregion


# region entities
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
        "on_touch": "stomp_kill",
        "damage": 1,
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
        "on_touch": "hurt_player",
        "damage": 1,
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
        "on_touch": "pickup",
    },
}

ENTITY_KIND_IDS = {kind: index for index, kind in enumerate(ENTITY_LIBRARY, start=1)}
ENTITY_KIND_NAMES = {index: kind for kind, index in ENTITY_KIND_IDS.items()}

# Entity state byte. Movement handlers can read/branch on this; it's part
# of the corruptible record on purpose (see FRAGILE ZONES above).
STATE_IDLE = 0
STATE_CHASE = 1
STATE_FLEE = 2
STATE_STUNNED = 3

# header: active(B), kind_id(B), state(B), pad(x)
# body:   x, y, vx, vy, speed_mult, target_offset  (6 doubles)
ENTITY_RECORD_FORMAT = ">BBBxdddddd"
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
        entity.state,
        entity.x,
        entity.y,
        entity.vx,
        entity.vy,
        entity.speed_mult,
        entity.target_offset,
    )


def read_entity_memory(slot):
    """Handy for a future debug/cheat overlay: reads a raw slot back out
    without going through the Entity object at all."""
    pointer = _entity_record_pointer(slot)
    return struct.unpack(ENTITY_RECORD_FORMAT, data[pointer:pointer + ENTITY_RECORD_SIZE])


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
    state: int = STATE_CHASE
    speed_mult: float = 1.0
    target_offset: float = 0.0

    def update(self, dt, player_x, player_y, tile_size):
        if not self.alive:
            return

        behavior = ENTITY_LIBRARY[self.kind]
        movement_name = behavior.get("movement", "ground_chaser")
        handler = MOVEMENT_HANDLERS.get(movement_name, move_wander)
        handler(self, dt, player_x, player_y)

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

    def draw(self, screen, camera_x, camera_y):
        if not self.alive:
            return
        pygame.draw.rect(screen, self.color, (self.x - camera_x, self.y - camera_y, self.width, self.height))


# --- Movement handlers ---------------------------------------------
# Pluggable, one function per "movement" value in ENTITY_LIBRARY. Each
# handler just needs to set entity.vx / entity.vy. speed_mult and
# target_offset live in the entity's raw memory record, so they're a
# built-in corruption surface (see FRAGILE ZONES).

def move_ground_chaser(entity, dt, player_x, player_y):
    dx = (player_x + entity.target_offset) - entity.x
    dist = max(1.0, abs(dx))
    entity.vx = (dx / dist) * entity.speed * entity.speed_mult


def move_flyer(entity, dt, player_x, player_y):
    dx = (player_x + entity.target_offset) - entity.x
    dy = player_y - entity.y
    dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
    entity.vx = (dx / dist) * entity.speed * entity.speed_mult
    entity.vy = (dy / dist) * entity.speed * entity.speed_mult * 0.75


def move_floater(entity, dt, player_x, player_y):
    dx = (player_x + entity.target_offset) - entity.x
    dy = player_y - entity.y
    dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
    entity.vx = (dx / dist) * entity.speed * entity.speed_mult * 0.5
    entity.vy = (dy / dist) * entity.speed * entity.speed_mult * 0.5


def move_wander(entity, dt, player_x, player_y):
    """Fallback used when an entity's movement name (or a corrupted
    state byte routed through here) doesn't match a known handler — it
    just drifts instead of crashing or freezing."""
    entity.vx = entity.speed * 0.25 * (1 if (entity.slot % 2 == 0) else -1)


MOVEMENT_HANDLERS = {
    "ground_chaser": move_ground_chaser,
    "flyer": move_flyer,
    "floater": move_floater,
}


# --- Touch scripts ---------------------------------------------------
# Each entity kind's "on_touch" names a handler here. Signature:
#   handler(entity, stomp) -> "none" | "kill_entity" | "damage_player" | "bounce"
# "damage_player" deals ENTITY_LIBRARY[kind]["damage"] (default 1) and
# grants a brief invincibility window - it no longer kills outright.
# Add new handlers and reference them from ENTITY_LIBRARY freely.

def touch_stomp_kill(entity, stomp):
    if stomp:
        return "bounce"
    return "damage_player"


def touch_hurt_player(entity, stomp):
    return "damage_player"


def touch_pickup(entity, stomp):
    return "kill_entity"


TOUCH_HANDLERS = {
    "stomp_kill": touch_stomp_kill,
    "hurt_player": touch_hurt_player,
    "pickup": touch_pickup,
}


def jump_velocity_for_bounce():
    return float(readData("jumpVelocity")) * 0.6


DASH_INVINCIBILITY_FRAMES = 20
HURT_INVINCIBILITY_FRAMES = 45  # i-frames after taking damage from a hazard/enemy
DASH_ACTIVE_FRAMES = 10      # how long the burst itself lasts
DASH_SPEED = 11.0            # constant horizontal speed during the burst
DASH_COOLDOWN_FRAMES = 55    # time before you can dash again


def player_is_invincible():
    return readData("invincibilityTimer") > 0


def damage_player(amount, knockback_dx=0.0, knockback_dy=-3.0):
    """Central place all non-lethal damage goes through: hazards and
    enemy touches call this instead of killing the player outright.
    Returns True if this hit brought health to 0 and triggered a full
    death (kill_player)."""
    if player_is_invincible():
        return False

    modifyData("invincibilityTimer", HURT_INVINCIBILITY_FRAMES)
    modifyData("playerXAcc", knockback_dx)
    modifyData("playerYAcc", knockback_dy)

    new_health = readData("playerHealth") - amount
    modifyData("playerHealth", new_health)

    if new_health <= 0:
        kill_player()
        return True
    return False


def resolve_entity_collisions(pre_collision_y_acc):
    player_w = int(readData("playerWidth"))
    player_h = int(readData("playerHeight"))
    player_rect = pygame.Rect(readData("playerX"), readData("playerY"), player_w, player_h)

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

        if outcome == "damage_player":
            damage = ENTITY_LIBRARY[entity.kind].get("damage", 1)
            push_dir = -1.0 if entity_rect.centerx > player_rect.centerx else 1.0
            if damage_player(damage, knockback_dx=push_dir * 3.0, knockback_dy=-3.0):
                return True

    return False


def kill_player():
    spawn_x = readData("playerSpawnX")
    spawn_y = readData("playerSpawnY")
    modifyData("playerX", float(spawn_x))
    modifyData("playerY", float(spawn_y))
    modifyData("playerXAcc", 0.0)
    modifyData("playerYAcc", 0.0)
    modifyData("onGround", False)
    modifyData("fastFall", True)
    modifyData("playerHealth", readData("playerMaxHealth"))
    reset_entities()


def reset_entities():
    entities.clear()
    clear_entity_memory()
    _attack_hit_slots.clear()
    for spawner in spawners:
        spawner.cooldown = 0.0


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
    spent: bool = False

    def update(self, dt, entity_list):
        if self.spent:
            return
        self.cooldown = max(0.0, self.cooldown - dt)
        if self.cooldown <= 0:
            self.spawn_batch(entity_list)
            self.spent = True

    def spawn_batch(self, entity_list):
        """Fires every entity in this spawner's one-time batch at once."""
        for _ in range(self.max_active):
            self.spawn_one(entity_list)

    def spawn_one(self, entity_list):
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
                state=STATE_CHASE,
            )
        )
        write_entity_memory(entity_list[-1])
# endregion


# region rooms (Tiled JSON)
#
# Export your map from the Tiled editor as JSON with:
#   - A tile layer named "Tiles": a flat "data" array of tile IDs, one
#     per cell, row-major. IMPORTANT simplification: this loader treats
#     each value in that array as a direct key into TILE_DEFINITIONS
#     (i.e. build your Tiled tileset with firstgid=1 and tiles in the
#     same order as TILE_DEFINITIONS' keys — tile index 1 = wall, etc).
#     0 means empty.
#   - An object layer named "Objects" containing:
#       * one object with type "spawn" -> player spawn point (x, y)
#       * objects with type "exit", each with custom properties
#         target_room (string, room key) and target_x/target_y (numbers)
#         -> touching one loads that room and drops the player there
#       * objects with type "spawner", with a custom property
#         entity_type (must match an ENTITY_LIBRARY key), and optionally
#         interval/max_active
#       * objects with type "camera_zone", each a rectangle with a
#         custom property mode = "bound" or "focus":
#           - "bound": while the player is standing inside this
#             rectangle, the camera follows them as usual but is clamped
#             to this rectangle instead of the whole room. Use this to
#             box the camera into a sub-area (a tight corridor, a boss
#             arena, etc).
#           - "focus": while the player is inside this rectangle, the
#             camera ignores the player entirely and centers on a fixed
#             point instead - optional numeric properties focus_x/
#             focus_y set that point; if omitted, it defaults to the
#             zone rectangle's own center.
#         If the player isn't standing inside ANY camera_zone, the
#         camera just follows them directly with no clamping at all -
#         including past the edges of the room's own tile grid. If the
#         player is inside more than one overlapping zone, the smallest
#         (by area) wins, so you can nest a tight zone inside a looser
#         one.
#   - Optional: "tilewidth"/"tileheight" on the map, and a "tilesetimage"
#     custom map property pointing at a sheet to slice per tile id
#     instead of using TILE_DEFINITIONS' flat render_color fallback.
#   - Optional: a boolean custom map property "liveMemoryRendering". See
#     render_tiles_live() below for what this actually does - it's a
#     per-room toggle, set from the map itself, so you can deliberately
#     choose which rooms are allowed to visually corrupt at their edges.
#
# ROOMS below is your room registry: key -> path to that room's JSON.
# Add new rooms here as you build them in Tiled.

ROOMS = {
    "room0": "rooms/room0.json",
}

room_exits = []       # list of {"rect": pygame.Rect, "target_room":..., "target_x":..., "target_y":...}
camera_zones = []     # list of {"rect": pygame.Rect, "mode": "bound"/"focus", "focus_x":..., "focus_y":...}
room_tileset_image = None
room_tile_size_from_sheet = None


def _load_tileset_image(path, tile_w, tile_h):
    global room_tileset_image, room_tile_size_from_sheet
    try:
        room_tileset_image = pygame.image.load(path).convert_alpha()
        room_tile_size_from_sheet = (tile_w, tile_h)
    except (pygame.error, FileNotFoundError):
        room_tileset_image = None
        room_tile_size_from_sheet = None


def get_tile_sprite(tile_id, target_size):
    """Returns a sub-surface for this tile id if a tileset image is
    loaded and has a matching tile, else None (caller falls back to a
    flat color). Always returned at target_size x target_size, scaled up
    or down from whatever pixel size the source tileset actually uses —
    so a 16x16 sheet still fills a 40x40 world tile correctly."""
    if room_tileset_image is None:
        return None
    tw, th = room_tile_size_from_sheet
    sheet_w = room_tileset_image.get_width()
    cols = max(1, sheet_w // tw)
    index = tile_id - 1
    if index < 0:
        return None
    col = index % cols
    row = index // cols
    rect = pygame.Rect(col * tw, row * th, tw, th)
    if rect.bottom > room_tileset_image.get_height():
        return None
    sprite = room_tileset_image.subsurface(rect)
    if (tw, th) != (target_size, target_size):
        sprite = pygame.transform.scale(sprite, (target_size, target_size))
    return sprite


def _properties_to_dict(properties_list):
    """Tiled exports custom properties (on the map, a layer, or an
    object) as a list of {"name", "type", "value"} dicts, not as plain
    top-level keys. This flattens that into a normal dict."""
    return {p["name"]: p["value"] for p in (properties_list or [])}


def load_room(room_key, entry_x=None, entry_y=None):
    """Loads a room by key from ROOMS into the shared tile memory region
    and (re)builds its spawners. entry_x/entry_y override the room's own
    spawn point (used when arriving through an exit)."""
    global spawners, entities, room_exits, camera_zones, room_tileset_image

    path = ROOMS.get(room_key)
    path = resolve_path(path)
    layout = None
    cols, rows = MAP_COLS, MAP_ROWS
    spawn_point = (100, 100)
    room_exits = []
    camera_zones = []
    room_tileset_image = None
    modifyData("liveTileRendering", False)

    if path and os.path.exists(path):
        try:
            with open(path, 'r') as f:
                room_json = json.load(f)

            cols = int(room_json.get("width", MAP_COLS))
            rows = int(room_json.get("height", MAP_ROWS))
            tile_w = int(room_json.get("tilewidth", TILE_SIZE))
            tile_h = int(room_json.get("tileheight", TILE_SIZE))

            tile_layer = next(
                (l for l in room_json.get("layers", []) if l.get("name") == "Tiles"),
                None,
            )
            layout = list(tile_layer["data"]) if tile_layer else None

            object_layer = next(
                (l for l in room_json.get("layers", []) if l.get("name") == "Objects"),
                None,
            )
            objects = object_layer.get("objects", []) if object_layer else []

            map_properties = _properties_to_dict(room_json.get("properties"))
            tileset_image_path = map_properties.get("tilesetimage")
            if tileset_image_path:
                _load_tileset_image(resolve_path(tileset_image_path), tile_w, tile_h)

            modifyData("liveTileRendering", bool(map_properties.get("liveMemoryRendering", False)))

            for obj in objects:
                obj_type = obj.get("type", "")
                props = _properties_to_dict(obj.get("properties"))

                if obj_type == "spawn":
                    spawn_point = (obj["x"], obj["y"])

                elif obj_type == "exit":
                    room_exits.append({
                        "rect": pygame.Rect(obj["x"], obj["y"], obj.get("width", tile_w), obj.get("height", tile_h)),
                        "target_room": props.get("target_room"),
                        "target_x": props.get("target_x", 100),
                        "target_y": props.get("target_y", 100),
                    })

                elif obj_type == "camera_zone":
                    camera_zones.append({
                        "rect": pygame.Rect(obj["x"], obj["y"], obj.get("width", tile_w), obj.get("height", tile_h)),
                        "mode": props.get("mode", "bound"),
                        "focus_x": props.get("focus_x"),
                        "focus_y": props.get("focus_y"),
                    })
                # "spawner" objects are handled below via build_spawners_from_tiles

        except (KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"Could not parse room '{room_key}' ({path}): {e}. Falling back to flat map.")
            layout = None

    if layout is None:
        print(f"Room '{room_key}' not found or invalid ({path}); using flat fallback map.")
        border_row = [1] * cols
        empty_row = [1] + [0] * (cols - 2) + [1]
        layout = []
        layout.extend(border_row)
        for _ in range(rows - 2):
            layout.extend(empty_row)
        layout.extend(border_row)

    modifyData("mapCols", cols)
    modifyData("mapRows", rows)
    # NOTE: intentionally not clearing the whole loadedTiles region first —
    # see FRAGILE ZONES at the top of this file.
    modifyData("loadedTiles", layout)

    px = entry_x if entry_x is not None else spawn_point[0]
    py = entry_y if entry_y is not None else spawn_point[1]
    modifyData("playerSpawnX", float(px))
    modifyData("playerSpawnY", float(py))
    modifyData("playerX", float(px))
    modifyData("playerY", float(py))
    modifyData("playerXAcc", 0.0)
    modifyData("playerYAcc", 0.0)
    modifyData("onGround", False)

    screen_width = int(readData("screenWidth"))
    screen_height = int(readData("screenHeight"))
    _, snapped_camera_y = compute_camera_target(screen_width, screen_height, readData("cameraLookahead"))
    modifyData("cameraY", snapped_camera_y)

    spawners = build_spawners_from_tiles(layout, cols, rows)
    entities = []
    clear_entity_memory()

    for room_index, key in enumerate(ROOMS):
        if key == room_key:
            modifyData("roomId", room_index)
            break


def build_spawners_from_tiles(tile_layout, cols, rows):
    spawners_list = []
    for index, tile_id in enumerate(tile_layout):
        tile_def = TILE_DEFINITIONS.get(tile_id)
        if tile_def is None or "spawner" not in tile_def:
            continue
        col = index % cols
        row = index // cols
        config = tile_def["spawner"]
        spawners_list.append(
            Spawner(
                spawner_id=len(spawners_list),
                x=col * TILE_SIZE,
                y=row * TILE_SIZE,
                entity_type=config["entity_type"],
                interval=config.get("interval", 2.0),
                max_active=config.get("max_active", 1),
                cooldown=config.get("interval", 2.0),
                width=40,
                height=40,
            )
        )
    return spawners_list


def compute_camera_target(screen_width, screen_height, camera_lookahead):
    """Figures out where the camera should be heading this frame, based
    on whichever camera_zone (if any) the player is currently standing
    in. No matching zone -> the camera just follows the player with no
    clamping at all, including past the room's own tile bounds."""
    player_w = int(readData("playerWidth"))
    player_h = int(readData("playerHeight"))
    px = readData("playerX")
    py = readData("playerY")
    player_rect = pygame.Rect(px, py, player_w, player_h)
    player_center_y = py + player_h * 0.5

    active_zone = None
    smallest_area = None
    for zone in camera_zones:
        if not zone["rect"].colliderect(player_rect):
            continue
        area = zone["rect"].width * zone["rect"].height
        if smallest_area is None or area < smallest_area:
            smallest_area = area
            active_zone = zone

    if active_zone is None:
        target_x = px - camera_lookahead
        target_y = player_center_y - screen_height * 0.5
        return target_x, target_y

    if active_zone["mode"] == "focus":
        focus_x = active_zone["focus_x"] if active_zone["focus_x"] is not None else active_zone["rect"].centerx
        focus_y = active_zone["focus_y"] if active_zone["focus_y"] is not None else active_zone["rect"].centery
        target_x = focus_x - screen_width * 0.5
        target_y = focus_y - screen_height * 0.5
        return target_x, target_y

    # mode == "bound": follow the player as usual, clamped to this zone's
    # own rectangle instead of the whole room.
    target_x = px - camera_lookahead
    target_y = player_center_y - screen_height * 0.5
    zone_rect = active_zone["rect"]
    min_x, max_x = zone_rect.left, max(zone_rect.left, zone_rect.right - screen_width)
    min_y, max_y = zone_rect.top, max(zone_rect.top, zone_rect.bottom - screen_height)
    target_x = max(min_x, min(max_x, target_x))
    target_y = max(min_y, min(max_y, target_y))
    return target_x, target_y


def render_tiles_live(screen, cam_x, cam_y, tile_size, cols, screen_width, screen_height):
    """Alternate tile renderer, toggled per-room via the liveMemoryRendering
    map property. Instead of drawing only the room's own bounded tile
    slice (current_map_bytes in the main loop), this computes an
    absolute offset into the whole shared `data` buffer for every tile
    the camera can currently see - using the exact same row*cols+col
    linear-memory formula the game already uses for real tile lookups.
    Inside the room's actual bounds this looks identical to normal
    rendering. Past its edges, it just keeps reading real bytes -
    other player state, entity records, whatever's physically next in
    memory - and reinterprets them as tile ids through the normal
    TILE_DEFINITIONS lookup. Most raw bytes won't match a known id and
    so render as nothing, but every so often one will happen to be
    1-6 and a "wall" or "platform" will materialize out of memory that
    was never actually part of the level. Offsets wrap with modulo
    instead of indexing out of range, so this can never crash - only
    ever show something weird."""
    map_start = int(readMemory("loadedTiles", "pointer"))
    buffer_len = len(data)

    first_col = int(cam_x // tile_size) - 1
    last_col = int((cam_x + screen_width) // tile_size) + 1
    first_row = int(cam_y // tile_size) - 1
    last_row = int((cam_y + screen_height) // tile_size) + 1

    for row in range(first_row, last_row + 1):
        for col in range(first_col, last_col + 1):
            array_index = row * cols + col
            raw_offset = (map_start + array_index) % buffer_len
            tile_id = data[raw_offset]

            if tile_id == EMPTY_TILE_ID:
                continue
            tile_def = TILE_DEFINITIONS.get(tile_id)
            if tile_def is None:
                continue

            screen_x = col * tile_size - cam_x
            screen_y = row * tile_size - cam_y

            sprite = get_tile_sprite(tile_id, tile_size)
            if sprite is not None:
                screen.blit(sprite, (screen_x, screen_y))
            else:
                size = tile_size - 1 if tile_def["grid_gap"] else tile_size
                pygame.draw.rect(screen, tile_def["render_color"], (screen_x, screen_y, size, size))


def check_room_exits():
    player_w = int(readData("playerWidth"))
    player_h = int(readData("playerHeight"))
    player_rect = pygame.Rect(readData("playerX"), readData("playerY"), player_w, player_h)
    for exit_def in room_exits:
        if player_rect.colliderect(exit_def["rect"]) and exit_def["target_room"] in ROOMS:
            load_room(exit_def["target_room"], exit_def["target_x"], exit_def["target_y"])
            return
# endregion


# region combat
#
# A basic melee weapon: pressing attack opens a brief active window
# (ATTACK_ACTIVE_FRAMES) during which a hitbox in front of the player
# damages anything it touches, then goes on cooldown. Each entity can
# only be hit once per swing (tracked in _attack_hit_slots) so a
# multi-health enemy doesn't melt in a single swing just because the
# hitbox overlapped it for several frames.

ATTACK_COOLDOWN_FRAMES = 20
ATTACK_ACTIVE_FRAMES = 6
ATTACK_RANGE = 30
WEAPON_DAMAGE = 1
WEAPON_KNOCKBACK = 3.5

_attack_hit_slots = set()  # entity slots already hit by the current swing


def get_attack_hitbox():
    player_w = int(readData("playerWidth"))
    player_h = int(readData("playerHeight"))
    px = readData("playerX")
    py = readData("playerY")
    facing = readData("playerFacing") or 1
    if facing >= 0:
        return pygame.Rect(px + player_w, py, ATTACK_RANGE, player_h)
    return pygame.Rect(px - ATTACK_RANGE, py, ATTACK_RANGE, player_h)


def trigger_attack():
    """Called on the attack input. Just opens the active swing window if
    the weapon isn't on cooldown - the actual hit detection happens every
    frame the window is open, via resolve_melee_attack()."""
    if readData("attackCooldown") > 0:
        return

    modifyData("attackCooldown", ATTACK_COOLDOWN_FRAMES)
    modifyData("attackActiveTimer", ATTACK_ACTIVE_FRAMES)
    _attack_hit_slots.clear()


def resolve_melee_attack():
    """Runs every frame the swing is active. Returns the current hitbox
    (for rendering) or None when the weapon isn't swinging."""
    if readData("attackActiveTimer") <= 0:
        return None

    modifyData("attackActiveTimer", readData("attackActiveTimer") - 1)
    hitbox = get_attack_hitbox()
    facing = readData("playerFacing") or 1

    for entity in entities:
        if not entity.alive or entity.slot in _attack_hit_slots:
            continue
        entity_rect = pygame.Rect(entity.x, entity.y, entity.width, entity.height)
        if not hitbox.colliderect(entity_rect):
            continue

        _attack_hit_slots.add(entity.slot)
        entity.health -= WEAPON_DAMAGE
        if entity.health <= 0:
            entity.alive = False
        else:
            entity.vx = facing * WEAPON_KNOCKBACK
            entity.vy = -1.5
        write_entity_memory(entity)

    return hitbox


# Side-ability framework: a resource meter (memory-mapped so it's
# inspectable/corruptible like everything else) and empty slots for you
# to design 1-3 Hollow-Knight-style spells/tools into.
ABILITY_RESOURCE_MAX = 100.0
ABILITY_RESOURCE_REGEN_PER_FRAME = 0.05

SIDE_ABILITY_SLOTS = {
    # "slot_name": {"cost": 25.0, "cooldown_frames": 30, "unlock": "someAbilityBit"},
}


def use_side_ability(slot_name):
    """TODO: implement your side abilities here. Framework checks the
    resource cost/ability unlock and does nothing else yet."""
    config = SIDE_ABILITY_SLOTS.get(slot_name)
    if config is None:
        return
    if readData("abilityResource") < config.get("cost", 0.0):
        return
    modifyData("abilityResource", readData("abilityResource") - config.get("cost", 0.0))
    # TODO: actual effect goes here.
# endregion


# region Level Initialization
MAP_COLS = 100
MAP_ROWS = 15
TILE_SIZE = 40

pygame.display.init()  # not pygame.init() - skips the audio mixer,
                       # which in a browser blocks on a user click even
                       # though this game doesn't use sound at all

modifyData("playerWidth", 25)
modifyData("playerHeight", 35)
modifyData("tileSize", TILE_SIZE)
modifyData("screenWidth", 800)
modifyData("screenHeight", 600)

modifyData("gravity", 0.3)
modifyData("jumpVelocity", -8.5)
modifyData("maxXAcc", 5.0)
modifyData("xAccStep", 0.4)
modifyData("frictionDivisor", 10.0)
modifyData("cameraLookahead", 400.0)
modifyData("cameraY", 0.0)
modifyData("invincibilityTimer", 0)
modifyData("coyoteTimer", 0)
modifyData("jumpBufferTimer", 0)
modifyData("wallSliding", False)
modifyData("playerFacing", 1)
modifyData("attackCooldown", 0)
modifyData("abilityResource", ABILITY_RESOURCE_MAX)
modifyData("dashTimer", 0)
modifyData("dashActiveTimer", 0)
modifyData("dashDirection", 1)
modifyData("dropThroughTimer", 0)
modifyData("airJumpsUsed", 0)
modifyData("playerMaxHealth", 3)
modifyData("playerHealth", 3)

# TODO: gate these behind real pickups once you design progression.
# Everything is unlocked by default right now so the movement kit is
# testable immediately.
for _ability in ABILITY_BITS:
    unlock_ability(_ability)

my_surface = pygame.Surface((int(readData("playerWidth")), int(readData("playerHeight"))))
my_surface.fill((255, 0, 0))

screen = pygame.display.set_mode((int(readData("screenWidth")), int(readData("screenHeight"))))

spawners = []
entities = []
load_room("room0")
#endregion


# region movement tuning
COYOTE_FRAMES = 6
JUMP_BUFFER_FRAMES = 6
JUMP_CUT_MULTIPLIER = 0.5
WALL_SLIDE_MAX_FALL_SPEED = 2.0
WALL_JUMP_PUSH = 4.5
CAMERA_VERTICAL_SMOOTH = 0.1  # how quickly the camera catches up vertically (0-1)
DROP_THROUGH_FRAMES = 12     # how long a one-way platform stays passable after pressing Down
MAX_AIR_JUMPS = 1            # extra jumps available once doubleJump is unlocked
# endregion


#region game loop
async def main():
    running = True
    clock = pygame.time.Clock()
    while running:
        screen.fill((0, 0, 0))

        tile_size = int(readData("tileSize"))
        player_w = int(readData("playerWidth"))
        player_h = int(readData("playerHeight"))
        gravity = readData("gravity")
        jump_velocity = readData("jumpVelocity")
        max_x_acc = readData("maxXAcc")
        x_acc_step = readData("xAccStep")
        friction_divisor = readData("frictionDivisor")
        camera_lookahead = readData("cameraLookahead")

        screen_height = int(readData("screenHeight"))
        target_camera_x, desired_camera_y = compute_camera_target(
            int(readData("screenWidth")), screen_height, camera_lookahead
        )
        modifyData("cameraX", target_camera_x)

        # Smoothed rather than snapping straight to the target, so jumps/falls
        # don't jerk the camera around.
        current_camera_y = readData("cameraY")
        modifyData(
            "cameraY",
            current_camera_y + (desired_camera_y - current_camera_y) * CAMERA_VERTICAL_SMOOTH,
        )

        jump_pressed_this_frame = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_w, pygame.K_SPACE):
                    jump_pressed_this_frame = True

                elif event.key == pygame.K_e and has_ability("dash"):
                    if readData("dashTimer") < 1:
                        modifyData("dashTimer", DASH_COOLDOWN_FRAMES)
                        modifyData("dashActiveTimer", DASH_ACTIVE_FRAMES)
                        modifyData("dashDirection", readData("playerFacing") or 1)
                        modifyData("invincibilityTimer", DASH_INVINCIBILITY_FRAMES)

                elif event.key == pygame.K_j:
                    trigger_attack()

                elif event.key in (pygame.K_DOWN, pygame.K_s) and readData("onGround"):
                    modifyData("dropThroughTimer", DROP_THROUGH_FRAMES)

            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_UP, pygame.K_w, pygame.K_SPACE):
                    # Variable jump height: releasing jump early cuts an
                    # upward velocity short instead of always doing a full arc.
                    if readData("playerYAcc") < 0:
                        modifyData("playerYAcc", readData("playerYAcc") * JUMP_CUT_MULTIPLIER)

        keys = pygame.key.get_pressed()

        dash_active_this_frame = readData("dashActiveTimer") > 0

        if dash_active_this_frame:
            # Dash overrides normal movement entirely: a flat, constant-speed
            # burst in whatever direction you were facing when you pressed
            # it - no friction decay, no direction drift.
            modifyData("playerXAcc", readData("dashDirection") * DASH_SPEED)
            modifyData("dashActiveTimer", readData("dashActiveTimer") - 1)
        else:
            # --- Horizontal Acceleration Input ---
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                if readData("playerXAcc") >= -max_x_acc:
                    modifyData("playerXAcc", readData("playerXAcc") - x_acc_step)
                modifyData("playerFacing", -1)
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                if readData("playerXAcc") <= max_x_acc:
                    modifyData("playerXAcc", readData("playerXAcc") + x_acc_step)
                modifyData("playerFacing", 1)
            else:
                modifyData("playerXAcc", readData("playerXAcc") - (readData("playerXAcc") / friction_divisor))

        modifyData("dashTimer", readData("dashTimer") - 1)
        modifyData("invincibilityTimer", max(0, readData("invincibilityTimer") - 1))
        modifyData("attackCooldown", max(0, readData("attackCooldown") - 1))
        modifyData("dropThroughTimer", max(0, readData("dropThroughTimer") - 1))
        modifyData(
            "abilityResource",
            min(ABILITY_RESOURCE_MAX, readData("abilityResource") + ABILITY_RESOURCE_REGEN_PER_FRAME),
        )

        # --- Coyote time / jump buffer bookkeeping ---
        if readData("onGround"):
            modifyData("coyoteTimer", COYOTE_FRAMES)
            modifyData("airJumpsUsed", 0)
        else:
            modifyData("coyoteTimer", max(0, readData("coyoteTimer") - 1))

        if jump_pressed_this_frame:
            modifyData("jumpBufferTimer", JUMP_BUFFER_FRAMES)
        else:
            modifyData("jumpBufferTimer", max(0, readData("jumpBufferTimer") - 1))

        # --- Vertical Acceleration Input (Jumping & Gravity) ---
        if dash_active_this_frame:
            modifyData("playerYAcc", 0.0)
        else:
            modifyData("playerYAcc", readData("playerYAcc") + gravity)

        can_jump = readData("onGround") or readData("coyoteTimer") > 0
        if readData("jumpBufferTimer") > 0 and can_jump:
            modifyData("playerYAcc", jump_velocity)
            modifyData("onGround", False)
            modifyData("fastFall", True)
            modifyData("jumpBufferTimer", 0)
            modifyData("coyoteTimer", 0)
        elif (
            readData("jumpBufferTimer") > 0
            and readData("wallSliding")
            and has_ability("wallJump")
        ):
            # Wall jump: consumes the buffered jump, pushes off the wall.
            push_dir = -1 if readData("playerXAcc") >= 0 else 1
            modifyData("playerXAcc", push_dir * WALL_JUMP_PUSH)
            modifyData("playerYAcc", jump_velocity)
            modifyData("jumpBufferTimer", 0)
            modifyData("wallSliding", False)
            modifyData("airJumpsUsed", 0)
        elif (
            readData("jumpBufferTimer") > 0
            and not readData("onGround")
            and has_ability("doubleJump")
            and readData("airJumpsUsed") < MAX_AIR_JUMPS
        ):
            # Double (or further) jump: same jump strength, usable mid-air
            # after coyote time has run out, a limited number of times.
            modifyData("playerYAcc", jump_velocity)
            modifyData("jumpBufferTimer", 0)
            modifyData("coyoteTimer", 0)
            modifyData("airJumpsUsed", readData("airJumpsUsed") + 1)

        if (keys[pygame.K_DOWN] or keys[pygame.K_s]) and not readData("onGround"):
            if readData("fastFall"):
                modifyData("playerYAcc", 4.0)
                modifyData("fastFall", False)

        if readData("onGround"):
            modifyData("dashTimer", readData("dashTimer") - 5)
            # Dash refresh on landing (only matters if it was mid-cooldown).
            if has_ability("dash") and readData("dashTimer") > 0:
                modifyData("dashTimer", min(readData("dashTimer"), 10))

        for spawner in spawners:
            spawner.update(1 / 60, entities)

        for entity in entities:
            entity.update(1 / 60, readData("playerX"), readData("playerY"), tile_size)

        for entity in entities:
            if not entity.alive:
                write_entity_memory(entity)
        entities[:] = [entity for entity in entities if entity.alive]

        # --- PHYSICS: X axis ---
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

        blocked_horizontal = bool(x_side and any(f["solid_sides"][x_side] for f in x_flags))

        if blocked_horizontal:
            modifyData("playerXAcc", 0.0)
        elif any(f["hazard"] for f in x_flags):
            hazard_flags = next(f for f in x_flags if f["hazard"])
            damage_player(
                hazard_flags.get("damage", 1),
                knockback_dx=-x_acc * 0.5,
                knockback_dy=-3.0,
            )
        elif any(f["ground"] for f in x_flags):
            modifyData("onGround", True)
        else:
            modifyData("playerX", new_px)

        # Wall slide: only while airborne, moving into a wall, and falling.
        if blocked_horizontal and not readData("onGround") and readData("playerYAcc") > 0:
            modifyData("wallSliding", True)
            if readData("playerYAcc") > WALL_SLIDE_MAX_FALL_SPEED:
                modifyData("playerYAcc", WALL_SLIDE_MAX_FALL_SPEED)
        else:
            modifyData("wallSliding", False)

        # --- PHYSICS: Y axis ---
        px = readData("playerX")
        py = readData("playerY")
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
            if flags.get("one_way") and readData("dropThroughTimer") > 0:
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

        # --- Entity touch scripts + melee attack + room transitions ---
        resolve_entity_collisions(y_acc)
        active_attack_hitbox = resolve_melee_attack()
        check_room_exits()

        # --- Rendering ---
        cam_x = readData("cameraX")
        cam_y = readData("cameraY")

        cols = int(readData("mapCols"))
        screen_width = int(readData("screenWidth"))
        screen_height = int(readData("screenHeight"))

        if readData("liveTileRendering"):
            render_tiles_live(screen, cam_x, cam_y, tile_size, cols, screen_width, screen_height)
        else:
            spointer = int(readMemory("loadedTiles", "pointer"))
            epointer = int(readMemory("loadedTiles", "epointer"))
            current_map_bytes = data[spointer:epointer]

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
                screen_y = world_y - cam_y

                if -tile_size <= screen_x <= screen_width and -tile_size <= screen_y <= screen_height:
                    sprite = get_tile_sprite(tile_id, tile_size)
                    if sprite is not None:
                        screen.blit(sprite, (screen_x, screen_y))
                    else:
                        size = tile_size - 1 if tile_def["grid_gap"] else tile_size
                        pygame.draw.rect(screen, tile_def["render_color"], (screen_x, screen_y, size, size))

        final_x = readData("playerX")
        final_y = readData("playerY")
        if player_is_invincible():
            flashing = (readData("invincibilityTimer") // 3) % 2 == 0
            my_surface.fill((255, 255, 255) if flashing else (255, 0, 0))
        else:
            my_surface.fill((255, 0, 0))
        screen.blit(my_surface, (final_x - cam_x, final_y - cam_y))

        for spawner in spawners:
            pygame.draw.rect(screen, (60, 60, 60), (spawner.x - cam_x, spawner.y - cam_y, spawner.width, spawner.height), 1)
        for entity in entities:
            entity.draw(screen, cam_x, cam_y)

        if active_attack_hitbox is not None:
            pygame.draw.rect(
                screen,
                (255, 255, 255),
                (
                    active_attack_hitbox.x - cam_x,
                    active_attack_hitbox.y - cam_y,
                    active_attack_hitbox.width,
                    active_attack_hitbox.height,
                ),
                2,
            )

        # --- HUD: health pips ---
        max_health = int(readData("playerMaxHealth"))
        current_health = int(readData("playerHealth"))
        for pip_index in range(max_health):
            pip_x = 20 + pip_index * 24
            pip_color = (220, 40, 40) if pip_index < current_health else (60, 20, 20)
            pygame.draw.circle(screen, pip_color, (pip_x, 20), 8)

        pygame.display.flip()
        clock.tick(60)
        await asyncio.sleep(0)

    pygame.quit()

asyncio.run(main())
#endregion