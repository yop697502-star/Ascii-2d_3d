ASCII 3D Renderer Library - Hướng Dẫn Sử Dụng Đầy Đủ
Mục Lục
Giới Thiệu
Cài Đặt
Kiến Trúc Hệ Thống
Hướng Dẫn Cơ Bản
Tính Năng Nâng Cao
API Reference
Ví Dụ Thực Tế
Tối Ưu Hóa
Giới Thiệu
ASCII 3D Renderer Library là một engine render 3D hoàn chỉnh sử dụng ký tự ASCII để hiển thị đồ họa. Library này mô phỏng pipeline render hiện đại (tương tự OpenGL 4.5) nhưng xuất ra màn hình dưới dạng text art.
Tính Năng Chính
✅ Render 3D đầy đủ: Camera, Mesh, Transform, Projection
✅ Lighting System: Point Light, Directional, Spot Light
✅ Shading: Lambert, Phong, PBR (Cook-Torrance)
✅ Shadow Mapping: Hard & Soft shadows (PCF)
✅ Post-Processing: SSAO, Bloom, Fog, Tonemapping
✅ Advanced Features: Compute Shaders, MSAA, Stencil Buffer
✅ 2D Overlay: GDI-like system cho HUD/UI
✅ Dual Pipeline: Fixed-Function + Core Profile
Cài Đặt
Yêu Cầu Hệ Thống
# Dependencies
numpy >= 1.20.0
Pillow >= 8.0.0  # Cho texture loading
Python >= 3.8
Cài Đặt
pip install numpy pillow
Import Library
from Ascii2 import *
Kiến Trúc Hệ Thống
1. Core Components
┌─────────────────────────────────────────┐
│         ASCII 3D Renderer               │
├─────────────────────────────────────────┤
│  Math Layer (Vec3, Mat4, Camera)        │
│  Mesh System (Cube, Sphere, OBJ)        │
│  Lighting (Point, Dir, Spot)            │
│  Rasterizer (Triangle, Line)            │
│  Shader System (Lambert, Phong, PBR)    │
│  Post-Processing (SSAO, Bloom, Fog)     │
└─────────────────────────────────────────┘
2. Pipeline Flow
Model Space → World Space → View Space → Clip Space → Screen Space
     ↓            ↓             ↓            ↓            ↓
  Transform    Camera      Projection   Culling    Rasterize
Hướng Dẫn Cơ Bản
1. Render Cảnh Đơn Giản
# Khởi tạo renderer
renderer = Renderer(width=80, height=40)

# Tạo camera
camera = Camera(
    position=Vec3(0, 2, 5),
    target=Vec3(0, 0, 0),
    aspect=80/40
)
renderer.camera = camera

# Tạo ánh sáng
light = Light(position=Vec3(5, 5, 5), intensity=2.0)
renderer.add_light(light)

# Tạo mesh
cube = Mesh.create_cube(size=2.0)

# Render loop
import time
angle = 0
while True:
    renderer.clear()
    
    # Xoay khối lập phương
    cube.transform = Mat4.rotation_y(angle)
    renderer.render(cube)
    
    # Hiển thị
    print("\033[H" + renderer.get_frame())
    
    angle += 0.05
    time.sleep(0.033)  # ~30 FPS
2. Tạo Các Hình Dạng Cơ Bản
# Khối lập phương
cube = Mesh.create_cube(size=1.0)

# Hình cầu
sphere = Mesh.create_sphere(radius=1.0, segments=16)

# Load từ file OBJ
mesh = Mesh.load_obj("model.obj")
3. Camera Control
# Perspective Camera
camera = Camera(
    position=Vec3(0, 5, 10),
    target=Vec3(0, 0, 0),
    up=Vec3(0, 1, 0),
    fov=math.radians(60),
    projection_type=ProjectionType.PERSPECTIVE
)

# Orthographic Camera
camera.projection_type = ProjectionType.ORTHOGRAPHIC

# Di chuyển camera quanh vật thể
radius = 5.0
angle = 0
camera.position = Vec3(
    math.cos(angle) * radius,
    2.0,
    math.sin(angle) * radius
)
Tính Năng Nâng Cao
1. Dual Pipeline System
# Khởi tạo ASCII Pipeline
pipeline = ASCIIPipeline(width=80, height=40)

# Sử dụng Fixed Function Pipeline (OpenGL 1.x style)
pipeline.set_pipeline(fixed_function=True)
pipeline.fixed_func.push_matrix()
pipeline.fixed_func.translate(0, 0, -5)
pipeline.fixed_func.rotate_y(angle)
pipeline.fixed_func.render_mesh(mesh)
pipeline.fixed_func.pop_matrix()

# Hoặc Core Profile (Modern OpenGL style)
pipeline.set_pipeline(fixed_function=False)
pipeline.render_mesh(mesh)
2. Advanced Lighting
# Point Light
point_light = Light(
    position=Vec3(0, 5, 0),
    color=Vec3(1, 1, 1),
    intensity=2.0
)

# Directional Light
dir_light = DirectionalLight(
    direction=Vec3(0, -1, 0),
    color=Vec3(1, 0.9, 0.8),
    intensity=1.5
)

# Spot Light
spot_light = SpotLight(
    position=Vec3(0, 5, 0),
    direction=Vec3(0, -1, 0),
    color=Vec3(1, 1, 1),
    intensity=2.0,
    cutoff=0.9  # cos(angle)
)

renderer.add_light(point_light)
3. PBR Shading
# Cook-Torrance PBR
pbr_shader = PBRShaderModule()

# Trong render loop
brightness = pbr_shader.calculate_pbr_intensity(
    normal=vertex_normal,
    view_dir=view_direction,
    light_dir=light_direction,
    roughness=0.5,
    metallic=0.2
)
4. Shadow Mapping
# Bật shadows
renderer.enable_shadows = True

# Tạo shadow map
shadow_map = ShadowMap(width=256, height=256)

# Soft shadows với PCF
shadow_factor = shadow_map.sample_pcf(x, y, depth, radius=2)
5. Post-Processing Effects
SSAO (Screen Space Ambient Occlusion)
# Trong renderer
def apply_ssao(self):
    radius = 2
    bias = 0.05
    
    for y in range(self.height):
        for x in range(self.width):
            occlusion = 0.0
            current_depth = self.depth_buffer[y, x]
            
            # Sample neighbors
            samples = [(x+2, y), (x-2, y), (x, y+1), (x, y-1)]
            for sx, sy in samples:
                if self.depth_buffer[sy, sx] < current_depth - bias:
                    occlusion += 1.0
            
            # Giảm độ sáng
            self.brightness_buffer[y, x] *= (1.0 - occlusion * 0.5)
Bloom
def apply_bloom(self, threshold=0.8, intensity=0.5):
    # Extract bright areas
    bright_pass = np.where(self.color_buffer > threshold, 
                           self.color_buffer, 0)
    
    # Blur
    blurred = self.gaussian_blur(bright_pass)
    
    # Additive blend
    self.color_buffer += blurred * intensity
Volumetric Lighting (God Rays)
god_rays = VolumetricLight(
    decay=0.95,
    exposure=0.4,
    density=0.8,
    samples=15
)

# Trong render loop
god_rays.apply(renderer, light_position, camera)
6. Compute Shaders
# Khởi tạo compute shader
compute = ComputeShaderASCII(width=80, height=40)

# Định nghĩa shader function
def edge_detect(x, y, framebuffer, shared_mem):
    # Sobel edge detection
    gx = (fb.brightness_buffer[y-1,x+1] + 
          2*fb.brightness_buffer[y,x+1] + 
          fb.brightness_buffer[y+1,x+1] -
          fb.brightness_buffer[y-1,x-1] - 
          2*fb.brightness_buffer[y,x-1] - 
          fb.brightness_buffer[y+1,x-1])
    
    magnitude = math.sqrt(gx*gx)
    shared_mem[(x, y)] = magnitude

# Dispatch
compute.dispatch(framebuffer, edge_detect)
compute.apply_shared_to_buffer(framebuffer)
7. MSAA (Multi-Sample Anti-Aliasing)
# Render ở resolution cao hơn
msaa_fb = MSAAFrameBuffer(width=80, height=40, samples=4)

# Render vào super-resolution buffer
# ... render code ...

# Downsample về resolution gốc
msaa_fb.resolve(target_framebuffer)
8. Stencil Buffer
# Khởi tạo stencil buffer
stencil = StencilBuffer(width=80, height=40)
stencil.enabled = True

# Thiết lập stencil test
stencil.func = StencilFunc.EQUAL
stencil.ref = 1
stencil.mask = 0xFF

# Thiết lập stencil operations
stencil.fail_op = StencilOp.KEEP
stencil.zfail_op = StencilOp.KEEP
stencil.zpass_op = StencilOp.REPLACE

# Sử dụng trong render
if stencil.test(x, y):
    # Render pixel
    stencil.update(x, y, True, True)
2D Overlay System (GDI-like)
1. Canvas & Device Context
# Tạo canvas
canvas = Canvas(width=80, height=40, bg=' ')

# Tạo device context
dc = DeviceContext(canvas)

# Vẽ text
dc.text_out(10, 5, "Hello ASCII!", align='left')

# Vẽ hình chữ nhật
dc.rect(10, 10, 20, 10, fill=True)

# Vẽ đường line với anti-aliasing
dc.line(0, 0, 40, 20)
2. Layers System
# Device context với layers
dc_layers = DeviceContextWithLayers(canvas)

# Tạo background
dc_layers.set_background_bitmap("bg.png")

# Thêm layers
layer1 = dc_layers.add_layer(alpha=1.0, visible=True)
layer2 = dc_layers.add_layer(alpha=0.7, visible=True)

# Vẽ lên layer
dc1 = DeviceContext(layer1.canvas)
dc1.text_out(0, 0, "Layer 1")

# Render tất cả layers
dc_layers.render_layers()
3. Brushes & Pens
# Custom pen
pen = Pen(ch='*', width=2, style='solid')
dc.select_pen(pen)

# Gradient brush
brush = Brush(gradient=[' ', '░', '▒', '▓', '█'])
dc.select_brush(brush)

# Pattern brush
pattern = [
    ['#', '.'],
    ['.', '#']
]
brush = Brush(pattern=pattern)
4. Bitmap Drawing
# Load và vẽ bitmap
dc.draw_bitmap(
    "image.png",
    dx=0, dy=0,
    scale_w=1.0, scale_h=1.0,
    use_gradient=True
)

# Bitmap với mask (transparency)
dc.draw_bitmap_masked(
    "sprite.png",
    dx=10, dy=10,
    mask_char=' '
)
Scene Management
Automatic Scene Manager
# Khởi tạo scene manager tự động
scene = ASCIISceneManager(width=120, height=40)

# Tùy chỉnh
scene.fps_limit = 30.0
scene.enable_fog = True
scene.fog_density = 0.05

# Chạy vòng lặp tự động
scene.run_scene()
Advanced Scene Manager
# Scene manager nâng cao với nhiều features
scene = AdvancedSceneManager(width=120, height=40)

# Thêm God Rays
scene.god_rays = VolumetricLight(
    decay=0.95,
    weight=0.4,
    samples=15
)

# Multi-Draw Indirect (render nhiều objects)
for i in range(20):
    pos = Vec3(random.uniform(-10, 10), 0, random.uniform(-10, 0))
    transform = Mat4.translation(pos.x, pos.y, pos.z)
    scene.mdi_buffer.add_command(cube_mesh, transform, radius=1.5)

# Chạy
scene.run_scene()
Particle System
# Khởi tạo particle system
particles = ParticleSystem(count=100, area=10.0)

# Update particles (trong loop)
particles.compute_update(dt=0.016, floor_y=0.0)

# Render particles
view_proj = camera.get_projection_matrix() @ camera.get_view_matrix()
particles.render(renderer, view_proj)
Query Objects & Statistics
# Occlusion Query
occlusion_query = QueryASCII(QueryType.SAMPLES_PASSED)
occlusion_query.begin()

# Render something
# ... render code ...

occlusion_query.end()
visible_samples = occlusion_query.get_result()

# Timer Query
timer_query = QueryASCII(QueryType.TIME_ELAPSED)
timer_query.begin()
# ... render code ...
timer_query.end()
render_time = timer_query.get_result()

# Pipeline Statistics
stats = PipelineStatistics()
print(stats.get_stats())
Advanced Techniques
1. Normal Mapping
normal_map = NormalMapModule()

# Tính tangent space
tangent = normal_map.calculate_tangent_space(
    v0, v1, v2,
    uv0, uv1, uv2
)

# Áp dụng normal map
perturbed_normal = normal_map.apply_normal_map(
    base_normal,
    normal_map_sample,
    tangent
)
2. Parallax Occlusion Mapping
# Tạo height map
height_map = HeightMap(width=64, height=64)

# Tính POM
pom = POMCalculator()
new_uv = pom.get_parallax_coords(
    current_uv=uv,
    view_dir_tangent=view_in_tangent_space,
    height_map=height_map,
    height_scale=0.1
)

# Sample texture với UV mới
color = texture.sample(new_uv.x, new_uv.y)
3. Tessellation
# CPU Tessellation
tessellator = Tessellator()

# Subdivide triangle
subdivided = tessellator.subdivide_triangle(
    triangle,
    level=2  # Số lần chia
)

# Adaptive Tessellation (dựa trên khoảng cách)
adaptive_tess = AdaptiveTessellator()
adaptive_tess.process(mesh, camera.position)
4. Geometry Shader Effects
# Extrude effect
geometry_shader = GeometryShaderCPU()
extruded_tris = geometry_shader.extrude_triangle(
    triangle,
    distance=0.1
)

# Explode effect
GeometryProcessor.explode_effect(mesh, magnitude=0.5)
5. Screen Space Reflections
# Khởi tạo SSR
ssr = SSRModule(step_size=0.5, max_steps=30)

# Áp dụng (sau khi render scene)
ssr.apply(renderer, camera)
6. Tiled Lighting (Forward+)
# Khởi tạo tiled lighting
tiled = TiledLighting(width=80, height=40)

# Cull lights vào tiles
tiled.cull_lights(lights, camera)

# Trong fragment shader
relevant_lights = tiled.get_lights_for_pixel(x, y)
for light in relevant_lights:
    # Tính lighting chỉ với lights trong tile
    pass
Performance Optimization
1. Frustum Culling
# Trong IndirectDrawBuffer
def frustum_cull(self, camera, draw_command):
    pos = draw_command.transform @ Vec3(0, 0, 0)
    to_obj = pos - camera.position
    dist = to_obj.length()
    
    # Distance culling
    if dist > camera.far + draw_command.bounding_radius:
        return False
    
    # FOV culling
    forward = (camera.target - camera.position).normalize()
    if forward.dot(to_obj.normalize()) < 0.5:
        return False
    
    return True
2. Hi-Z Occlusion Culling
# Tạo Hi-Z buffer
hiz = HiZBuffer(renderer.framebuffer.depth_buffer)

# Kiểm tra occlusion
is_occluded = hiz.is_occluded(min_x, min_y, max_x, max_y, min_z)
if is_occluded:
    # Skip rendering
    continue
3. Instancing
# Render nhiều objects giống nhau
instancer = Instancer()

transforms = [
    Mat4.translation(i, 0, 0) for i in range(10)
]

instancer.render_instances(renderer, mesh, transforms)
4. Level of Detail (LOD)
def get_lod_mesh(distance):
    if distance < 5.0:
        return high_detail_mesh
    elif distance < 10.0:
        return medium_detail_mesh
    else:
        return low_detail_mesh

mesh = get_lod_mesh((obj_pos - camera.position).length())
Complete Example: 3D Game Scene
import time
import math
from Ascii2 import *

class Game3D:
    def __init__(self):
        # Setup renderer
        self.scene = AdvancedSceneManager(width=120, height=40)
        self.scene.fps_limit = 30
        
        # Create world
        self.ground = Mesh.create_cube(size=20.0)
        self.ground.transform = Mat4.translation(0, -1, 0) @ Mat4.scale(10, 0.1, 10)
        
        # Create player
        self.player = Mesh.create_sphere(radius=0.5, segments=8)
        self.player_pos = Vec3(0, 0, 0)
        
        # Enemies
        self.enemies = []
        for i in range(5):
            enemy = Mesh.create_cube(size=0.8)
            pos = Vec3(i * 3 - 6, 0, -5)
            self.enemies.append((enemy, pos))
        
        # Particles
        self.particles = ParticleSystem(count=50, area=10)
        
        # Post-processing
        self.scene.god_rays = VolumetricLight(decay=0.95, samples=15)
        
    def update(self, dt):
        # Update player
        self.player_pos.x += math.sin(time.time()) * dt
        self.player.transform = Mat4.translation(
            self.player_pos.x,
            self.player_pos.y,
            self.player_pos.z
        )
        
        # Update enemies
        for enemy, pos in self.enemies:
            pos.z += dt * 2
            if pos.z > 5:
                pos.z = -10
            enemy.transform = Mat4.translation(pos.x, pos.y, pos.z)
        
        # Update particles
        self.particles.compute_update(dt)
        
        # Update camera
        self.scene.camera.position = Vec3(
            self.player_pos.x,
            5,
            self.player_pos.z + 10
        )
        self.scene.camera.target = self.player_pos
        
    def render(self):
        # Clear
        self.scene.pipeline.clear()
        
        # Render objects
        self.scene.pipeline.render_mesh(self.ground)
        self.scene.pipeline.render_mesh(self.player)
        
        for enemy, _ in self.enemies:
            self.scene.pipeline.render_mesh(enemy)
        
        # Render particles
        vp = (self.scene.camera.get_projection_matrix() @ 
              self.scene.camera.get_view_matrix())
        self.particles.render(self.scene.pipeline.renderer, vp)
        
        # Post-processing
        self.scene.god_rays.apply(
            self.scene.pipeline.renderer,
            self.scene.light.position,
            self.scene.camera
        )
        
        # 2D HUD
        self._render_hud()
        
    def _render_hud(self):
        # FPS counter
        fps = 1.0 / self.scene.pipeline.renderer.frame_time
        hud_layer = self.scene.hud_layer
        dc = DeviceContext(hud_layer.canvas)
        dc.text_out(1, 1, f"FPS: {fps:.1f}", align='left')
        dc.text_out(1, 2, f"Pos: {self.player_pos.x:.1f}, {self.player_pos.z:.1f}")
        
    def run(self):
        last_time = time.time()
        while True:
            try:
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                
                self.update(dt)
                self.render()
                
                # Display
                print("\033[H" + self.scene.pipeline.get_frame())
                
                time.sleep(max(0, 1/30 - dt))
                
            except KeyboardInterrupt:
                break

# Run game
if __name__ == "__main__":
    game = Game3D()
    game.run()
Troubleshooting
Common Issues
Framerate quá thấp
Giảm resolution: Renderer(width=60, height=30)
Tắt post-processing: renderer.enable_fog = False
Giảm số lượng lights
Sử dụng frustum culling
Ký tự bị méo
Đảm bảo terminal hỗ trợ Unicode
Sử dụng font monospace
Điều chỉnh aspect ratio camera
Shadow artifacts
Tăng shadow map resolution
Điều chỉnh bias value
Sử dụng PCF filtering
Z-fighting
Điều chỉnh near/far planes của camera
Tăng precision của depth buffer
API Quick Reference
Core Classes
Vec3(x, y, z) - 3D Vector
Mat4 - 4x4 Transformation Matrix
Camera - Camera với perspective/orthographic
Mesh - 3D Mesh container
Renderer - Main renderer
Lighting
Light - Point light
DirectionalLight - Directional light
SpotLight - Spotlight
ASCIILight - ASCII-specific light
Effects
VolumetricLight - God rays
SSRModule - Screen space reflections
ShadowMap - Shadow mapping
StencilBuffer - Stencil operations
Systems
ASCIIPipeline - Dual pipeline manager
ASCIISceneManager - Automatic scene management
ParticleSystem - GPU particle simulation
ComputeShaderASCII - Compute shader
2D
Canvas - 2D drawing surface
DeviceContext - GDI-like drawing
Pen, Brush, Font - Drawing tools
Conclusion
Library này cung cấp một pipeline render 3D đầy đủ với khả năng mở rộng cao. Bạn có thể sử dụng nó để:
Tạo games ASCII 3D
Visualize dữ liệu khoa học
Tạo demo effects
Học về computer graphics
Tạo terminal-based applications
![ASCII 3D Demo](Demo.mp4)
