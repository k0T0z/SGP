import moderngl
import moderngl_window as mglw
import numpy as np
from PIL import Image
import os
import sys

# Shared by the live window and the offscreen renderer the genetic algorithm samples with.
VERT_SHADER = """
    #version 330
    in vec2 in_vert;
    in vec2 in_texcoord;
    out vec2 FragCoord;
    void main() {
        gl_Position = vec4(in_vert, 0.0, 1.0);
        FragCoord = in_texcoord;
    }
"""

QUAD_VERTICES = np.array([
    # x, y, u, v
    -1.0,  1.0, 0.0, 1.0,
    -1.0, -1.0, 0.0, 0.0,
     1.0,  1.0, 1.0, 1.0,
     1.0, -1.0, 1.0, 0.0,
], dtype='f4')


def _vertex_array(ctx, program, vbo):
    """Bind the quad, skipping attributes the driver optimised away.

    A generated shader that never reads FragCoord lets the compiler drop ``in_texcoord``, and
    moderngl raises when asked to bind an attribute the program does not have.
    """
    if program.get('in_texcoord', None) is not None:
        return ctx.vertex_array(program, [(vbo, '2f 2f', 'in_vert', 'in_texcoord')])
    return ctx.vertex_array(program, [(vbo, '2f', 'in_vert')])


class ShaderRenderer(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "SGP - Shader View (Isolated Widgets)"
    window_size = (1024, 512)
    aspect_ratio = 2.0
    resizable = True

    # These will be set before running the window
    initial_shader = ""
    initial_target = None

    # Set by ``run_live``: the size candidates are scored at, and the driver ``on_render`` steps
    # once per frame. The live driver lives in a dict because a plain function kept as a class
    # attribute would bind ``self`` when read back.
    live_offscreen_size = None
    live = None

    def __init__(self, **kwargs):
        # Runs with a window (moderngl_window supplies wnd/ctx) or without one, in which case
        # the caller passes a standalone ctx and an ``offscreen_size`` to read pixels back from.
        self.wnd = kwargs.get('wnd')
        self.ctx = kwargs.get('ctx')
        self.offscreen_size = kwargs.get('offscreen_size') or ShaderRenderer.live_offscreen_size

        if self.wnd:
            super().__init__(**kwargs)
        
        if not self.ctx and self.wnd:
            self.ctx = self.wnd.ctx
            
        self.quad_vertices = QUAD_VERTICES

        self.vbo = self.ctx.buffer(self.quad_vertices)
        
        # Shared vertex shader that provides FragCoord to the fragment shader
        self.vert_shader = VERT_SHADER

        # Simple shader for the target image widget
        self.image_frag = """
            #version 330
            out vec4 f_color;
            in vec2 FragCoord;
            uniform sampler2D target_image;
            void main() {
                f_color = texture(target_image, FragCoord);
            }
        """
        
        self.shader_prog = None
        self.shader_vao = None
        self.image_prog = self.ctx.program(vertex_shader=self.vert_shader, fragment_shader=self.image_frag)
        self.image_vao = self.ctx.vertex_array(self.image_prog, [
            (self.vbo, '2f 2f', 'in_vert', 'in_texcoord')
        ])
        
        self.target_texture = None
        self.target_pixels = None

        # Only allocated when there is no window: it is what the offscreen renderers read from.
        self.fbo = None
        if self.offscreen_size:
            width, height = self.offscreen_size
            self.fbo = self.ctx.framebuffer([self.ctx.texture((width, height), 4)])
        
        if ShaderRenderer.initial_shader:
            self.update_shader(ShaderRenderer.initial_shader)
            
        if ShaderRenderer.initial_target:
            self.load_target_image(ShaderRenderer.initial_target)

    def update_shader(self, glsl):
        """Compiles the raw generated shader for the left widget."""
        # Prepend version if missing
        if not glsl.strip().startswith("#version"):
            glsl = "#version 330\n" + glsl

        # The genetic algorithm recompiles this thousands of times, so drop the previous
        # program/VAO instead of letting the driver accumulate them.
        if self.shader_vao is not None:
            self.shader_vao.release()
            self.shader_vao = None
        if self.shader_prog is not None:
            self.shader_prog.release()
            self.shader_prog = None

        try:
            # We use the raw GLSL directly. The generator already provides:
            # - in vec2 FragCoord;
            # - uniform float uTime;
            # - out vec4 FragColor;
            # - void main() { ... }
            self.shader_prog = self.ctx.program(vertex_shader=self.vert_shader, fragment_shader=glsl)
            self.shader_vao = _vertex_array(self.ctx, self.shader_prog, self.vbo)
        except Exception as e:
            print(f"Failed to compile generated shader: {e}")
            # Fallback to error shader
            error_frag = """
                #version 330
                in vec2 FragCoord;
                out vec4 FragColor;
                void main() { FragColor = vec4(1.0, 0.0, 0.0, 1.0); }
            """
            self.shader_prog = self.ctx.program(vertex_shader=self.vert_shader, fragment_shader=error_frag)
            self.shader_vao = _vertex_array(self.ctx, self.shader_prog, self.vbo)

    def load_target_image(self, path, size=None):
        """Load the target, upload it for the window and keep its pixels for comparison.

        ``size`` is the ``(width, height)`` the pixels are resized to. It defaults to the
        offscreen framebuffer size, so a render and the target it is compared against always
        come back at the same resolution; without an offscreen size the image is left as is.
        """
        if not os.path.exists(path):
            print(f"Image not found: {path}")
            return

        size = size or self.offscreen_size
        img = Image.open(path).convert('RGBA')
        if size is not None and img.size != size:
            img = img.resize(size, Image.LANCZOS)

        # Kept top down, matching the flipped readback of an offscreen render.
        self.target_pixels = np.asarray(img, dtype=np.uint8)
        # GL samples with v=0 at the bottom, so the texture is uploaded the other way up.
        flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
        self.target_texture = self.ctx.texture(img.size, 4, flipped.tobytes())

    def draw_shader(self, time=0.0):
        """Draw the generated shader over the current viewport."""
        if not (self.shader_prog and self.shader_vao):
            return
        if 'uTime' in self.shader_prog:
            self.shader_prog['uTime'].value = time
        self.shader_vao.render(moderngl.TRIANGLE_STRIP)

    def draw_target(self, unit=0):
        """Draw the target image over the current viewport."""
        if not self.target_texture:
            return
        self.target_texture.use(unit)
        if 'target_image' in self.image_prog:
            self.image_prog['target_image'].value = unit
        self.image_vao.render(moderngl.TRIANGLE_STRIP)

    def _advance_ga(self):
        """Drive the live run one generation and show its current best."""
        live = ShaderRenderer.live
        if live["done"]:
            return
        glsl = live["step"](self)
        if glsl is None:
            # The run is over; the winner already compiled stays on screen.
            live["done"] = True
            return
        self.update_shader(glsl)

    def on_render(self, time, frame_time):
        if ShaderRenderer.live is not None:
            # Scoring binds the offscreen framebuffer, so take the window's back before drawing.
            self._advance_ga()
            self.ctx.screen.use()

        self.ctx.clear(0.1, 0.1, 0.1)
        
        # Get current window size
        width, height = self.wnd.size if self.wnd else self.fbo.size
        half_width = width // 2
        
        # --- Left Widget: Generated Shader ---
        self.ctx.viewport = (0, 0, half_width, height)
        self.draw_shader(time)
            
        # --- Right Widget: Target Image ---
        self.ctx.viewport = (half_width, 0, half_width, height)
        self.draw_target()

    def render_pixels(self, time=0.0):
        """Render the generated shader alone and return a top-down (H, W, 4) uint8 array.

        Needs the offscreen framebuffer, i.e. a renderer built with ``offscreen_size``.
        """
        if self.fbo is None:
            raise RuntimeError("render_pixels() needs an offscreen_size; this renderer has a window.")
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.viewport = (0, 0, self.fbo.width, self.fbo.height)
        self.draw_shader(time)
        return self._read_pixels()

    def render_scene_pixels(self, time=0.0):
        """Render the full two-widget layout offscreen and return it top-down."""
        if self.fbo is None:
            raise RuntimeError("render_scene_pixels() needs an offscreen_size; this renderer has a window.")
        self.fbo.use()
        self.on_render(time, 0.0)
        return self._read_pixels()

    def _read_pixels(self):
        """Read the offscreen framebuffer as a top-down (H, W, 4) uint8 array."""
        data = self.fbo.read(components=4, dtype='f1')
        image = np.frombuffer(data, dtype=np.uint8).reshape(self.fbo.height, self.fbo.width, 4)
        # moderngl reads bottom-up; flip so row 0 is the top like a normal image file.
        return image[::-1].copy()


def create_context():
    """A windowless GL context, for running the renderer offscreen."""
    return moderngl.create_standalone_context()


def run_live(step, target_image_path=None, resolution=256):
    """Show the two-widget window and let ``step`` drive the run, one call per frame.

    ``step`` is handed this renderer on the GL thread with the window's context current, so a
    candidate can be scored offscreen and read back exactly as in a headless run. It returns the
    GLSL of the individual to display, or ``None`` once the run is over, after which the window
    keeps showing the last result until it is closed.
    """
    ShaderRenderer.live_offscreen_size = (resolution, resolution)
    ShaderRenderer.initial_shader = ""
    ShaderRenderer.initial_target = target_image_path
    ShaderRenderer.live = {"step": step, "done": False}

    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        mglw.run_window_config(ShaderRenderer)
    finally:
        sys.argv = original_argv
        ShaderRenderer.live = None
        ShaderRenderer.live_offscreen_size = None
