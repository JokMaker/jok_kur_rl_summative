"""
OpenGLRenderer
==============

Renders the MicroloanAllocationEnv as a 3D "financial village": each member
is drawn as a building whose height encodes their savings balance and whose
color encodes their repayment health (green = healthy, red = risky). A
central tower represents the fund itself, growing or shrinking with the
fund's liquidity ratio. The most recently processed member pulses gold
when their loan is approved, so the agent's decisions are visible in real
time as the simulation runs.

Uses GLFW + PyOpenGL (an advanced rendering stack, distinct from simple
2D shape-drawing) so the agent's behavior can be inspected visually.
"""

from __future__ import annotations

import math

try:
    import glfw
    from OpenGL.GL import *
    from OpenGL.GLU import *
    _OPENGL_AVAILABLE = True
except Exception:  # pragma: no cover - headless / missing GL dev libs
    _OPENGL_AVAILABLE = False


class OpenGLRenderer:
    def __init__(self, n_members: int, width: int = 900, height: int = 650):
        self.n_members = n_members
        self.width = width
        self.height = height
        self.pulse = 0.0
        self._window = None

        if not _OPENGL_AVAILABLE:
            print(
                "[rendering] OpenGL/GLFW not available in this environment - "
                "running without a visible window. Training is unaffected."
            )
            return

        if not glfw.init():
            print("[rendering] glfw.init() failed - continuing without a window.")
            return

        self._window = glfw.create_window(
            width, height, "Microloan Allocation Agent - Financial Village", None, None
        )
        if not self._window:
            glfw.terminate()
            self._window = None
            return

        glfw.make_context_current(self._window)
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.53, 0.72, 0.90, 1.0)  # sky blue

    # ------------------------------------------------------------------ #
    def _setup_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(50, self.width / self.height, 0.1, 200.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 22, 34, 0, 0, 0, 0, 1, 0)

    def _draw_box(self, x, y, z, w, h, d, color):
        glPushMatrix()
        glTranslatef(x, y, z)
        glColor3f(*color)
        hw, hh, hd = w / 2, h / 2, d / 2
        verts = [
            (-hw, 0, -hd), (hw, 0, -hd), (hw, h, -hd), (-hw, h, -hd),
            (-hw, 0, hd), (hw, 0, hd), (hw, h, hd), (-hw, h, hd),
        ]
        faces = [
            (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
            (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4),
        ]
        glBegin(GL_QUADS)
        for face in faces:
            for idx in face:
                glVertex3f(*verts[idx])
        glEnd()
        glPopMatrix()

    def draw(self, env):
        if self._window is None or not _OPENGL_AVAILABLE:
            return
        if glfw.window_should_close(self._window):
            self.close()
            return

        glfw.make_context_current(self._window)
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._setup_projection()

        # ground plane
        self._draw_box(0, -0.1, 0, 30, 0.1, 30, (0.55, 0.45, 0.30))

        # central fund tower - height scaled by liquidity ratio
        liquidity_ratio = env.fund_capital / (100_000.0 * 1.5)
        fund_height = max(0.5, liquidity_ratio * 10)
        fund_color = (
            (0.85, 0.65, 0.10) if liquidity_ratio > 0.4 else (0.75, 0.20, 0.20)
        )
        self._draw_box(0, 0, 0, 2.5, fund_height, 2.5, fund_color)

        # member buildings arranged in a ring around the fund tower
        n = self.n_members
        radius = 12
        for i, member in enumerate(env.members):
            angle = 2 * math.pi * i / n
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            height = max(0.3, member.savings_balance / 1000.0)
            health = member.repayment_score  # 0..1
            color = (1 - health, health, 0.15)

            if i == env.current_member_idx and env.last_outcome is not None:
                pulse = 0.5 + 0.5 * math.sin(self.pulse)
                if env.last_outcome == "repaid":
                    color = (pulse * 0.2, 1.0, pulse * 0.2)
                elif env.last_outcome == "defaulted":
                    color = (1.0, pulse * 0.2, pulse * 0.2)
                else:
                    color = (0.6, 0.6, 0.6)

            self._draw_box(x, 0, z, 1.2, height, 1.2, color)

        self.pulse += 0.3
        glfw.swap_buffers(self._window)
        glfw.poll_events()

    def close(self):
        if _OPENGL_AVAILABLE and self._window is not None:
            glfw.destroy_window(self._window)
            glfw.terminate()
            self._window = None
