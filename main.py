import pygame
import sys
import struct
import json
import numpy as np

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
}

SOURCE_COLOR_TO_TILE_ID = {
    tile_def["source_color"]: tile_id
    for tile_id, tile_def in TILE_DEFINITIONS.items()
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

# --- Push configuration into the memory-mapped data array ---
# These used to be plain Python constants; now they live in `data`
# alongside everything else, so they can be inspected/modified through
# the same readMemory/modifyData interface as playerX, cameraX, etc.
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
    print(readData("dashTimer"))
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
        modifyData("playerX", float(readData("playerSpawn")[0]))
        modifyData("playerY", float(readData("playerSpawn")[1]))
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

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
#endregion