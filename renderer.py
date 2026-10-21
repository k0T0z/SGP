import moderngl
import moderngl_window as mglw
import numpy as np
from PIL import Image
import os
import sys

class ShaderRenderer(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "SGP - Shader Preview (Isolated Widgets)"
    window_size = (1024, 512)
    aspect_ratio = 2.0
    resizable = True

    # These will be set before running the window
    initial_shader = ""
    initial_target = None

    def __init__(self, **kwargs):
        # When running headless, we might not have a window object
        self.wnd = kwargs.get('wnd')
        self.ctx = kwargs.get('ctx')
        
        if self.wnd:
            super().__init__(**kwargs)
        
        if not self.ctx and self.wnd:
            self.ctx = self.wnd.ctx
            
        self.quad_vertices = np.array([
            # x, y, u, v
            -1.0,  1.0, 0.0, 1.0,
            -1.0, -1.0, 0.0, 0.0,
             1.0,  1.0, 1.0, 1.0,
             1.0, -1.0, 1.0, 0.0,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(self.quad_vertices)
        
        # Shared vertex shader that provides FragCoord to the fragment shader
        self.vert_shader = """
            #version 330
            in vec2 in_vert;
            in vec2 in_texcoord;
            out vec2 FragCoord;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
                FragCoord = in_texcoord;
            }
        """

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
        
        if ShaderRenderer.initial_shader:
            self.update_shader(ShaderRenderer.initial_shader)
            
        if ShaderRenderer.initial_target:
            self.load_target_image(ShaderRenderer.initial_target)

    def update_shader(self, glsl):
        """Compiles the raw generated shader for the left widget."""
        # Prepend version if missing
        if not glsl.strip().startswith("#version"):
            glsl = "#version 330\n" + glsl
            
        try:
            # We use the raw GLSL directly. The generator already provides:
            # - in vec2 FragCoord;
            # - uniform float uTime;
            # - out vec4 FragColor;
            # - void main() { ... }
            self.shader_prog = self.ctx.program(vertex_shader=self.vert_shader, fragment_shader=glsl)
            self.shader_vao = self.ctx.vertex_array(self.shader_prog, [
                (self.vbo, '2f 2f', 'in_vert', 'in_texcoord')
            ])
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
            self.shader_vao = self.ctx.vertex_array(self.shader_prog, [
                (self.vbo, '2f 2f', 'in_vert', 'in_texcoord')
            ])

    def load_target_image(self, path):
        if not os.path.exists(path):
            print(f"Image not found: {path}")
            return
            
        img = Image.open(path).convert('RGB')
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        self.target_texture = self.ctx.texture(img.size, 3, img.tobytes())

    def on_render(self, time, frame_time):
        self.ctx.clear(0.1, 0.1, 0.1)
        
        # Get current window size
        width, height = self.wnd.size if self.wnd else (1024, 512)
        half_width = width // 2
        
        # --- Left Widget: Generated Shader ---
        self.ctx.viewport = (0, 0, half_width, height)
        if self.shader_prog and self.shader_vao:
            if 'uTime' in self.shader_prog:
                self.shader_prog['uTime'].value = time
            self.shader_vao.render(moderngl.TRIANGLE_STRIP)
            
        # --- Right Widget: Target Image ---
        self.ctx.viewport = (half_width, 0, half_width, height)
        if self.target_texture:
            self.target_texture.use(0)
            if 'target_image' in self.image_prog:
                self.image_prog['target_image'].value = 0
            self.image_vao.render(moderngl.TRIANGLE_STRIP)

def run_preview(shader_code, target_image_path=None, headless=False):
    if headless:
        print("Running in headless mode...")
        ctx = moderngl.create_standalone_context()
        width, height = 1024, 512
        fbo = ctx.framebuffer(color_attachments=[ctx.texture((width, height), 4)])
        fbo.use()
        
        renderer_inst = ShaderRenderer(ctx=ctx, wnd=None)
        renderer_inst.update_shader(shader_code)
        if target_image_path:
            renderer_inst.load_target_image(target_image_path)
        
        # Manually trigger render with fixed size since wnd is None
        renderer_inst.on_render(0.0, 0.0)
        
        data = fbo.read(components=4)
        img = Image.frombytes('RGBA', (width, height), data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save('output.png')
        print("Headless render saved to output.png")
    else:
        ShaderRenderer.initial_shader = shader_code
        ShaderRenderer.initial_target = target_image_path
        
        original_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            mglw.run_window_config(ShaderRenderer)
        finally:
            sys.argv = original_argv
