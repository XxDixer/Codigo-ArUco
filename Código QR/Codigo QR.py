#!/usr/bin/env python3
"""
Realidad aumentada — marcador ArUco + OpenGL (GLFW) + Interaccion con Mano.

FIXES aplicados:
  - GRAB_RADIUS_PX corregido de 0 a 90 px  (bug: nunca se podia agarrar)
  - Cambio de modo teapot/sphere ya no crashea: se usa una bandera
    _pending_glut_init que se procesa al inicio del siguiente frame,
    ANTES de cualquier llamada de dibujado, en el hilo principal de OpenGL.

Controles:
  T         alternar tetera / esfera
  +/-       escala del modelo 3D
  ESC  Q    salir
  Pinza     agarrar / soltar el objeto con la mano
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import (
    GLU_FILL,
    gluNewQuadric,
    gluQuadricDrawStyle,
    gluSphere,
)

# ---------------------------------------------------------------------------
# MediaPipe (opcional)
# ---------------------------------------------------------------------------
try:
    import mediapipe as mp
    _mp_hands   = mp.solutions.hands
    _mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print(
        "[ADVERTENCIA] mediapipe no encontrado — interaccion con mano desactivada.\n"
        "              Instala con:  pip install mediapipe",
        file=sys.stderr,
    )

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
CAMERA_INDEX    = 0
MARKER_LENGTH_M = 0.10
ARUCO_DICT      = cv2.aruco.DICT_4X4_50
MARKER_ID       = 0
MODEL_SCALE     = 0.04
OBJECT_MODE     = "sphere"      # "teapot" | "sphere"
WINDOW_TITLE    = "RA: ArUco + Mano  |  T=objeto  ESC=salir  Pinza=agarrar"
ZNear, ZFar     = 0.01, 100.0

# FIX #2: valor corregido de 0 -> 90 px
PINCH_THRESHOLD = 0.20
GRAB_RADIUS_PX  = 1000

SPRING_K = 14.0
SPRING_D =  7.0

SCRIPT_DIR = Path(__file__).resolve().parent
CALIB_NPZ  = SCRIPT_DIR / "camera_ar.npz"

_pending_glut_init = False


# ===========================================================================
# Funciones del codigo base
# ===========================================================================

def default_camera_matrix(width: int, height: int) -> np.ndarray:
    f = float(max(width, height))
    cx, cy = width / 2.0, height / 2.0
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)


def load_calibration(width: int, height: int):
    if CALIB_NPZ.is_file():
        data = np.load(CALIB_NPZ)
        return data["camera_matrix"], data["dist_coeffs"]
    return default_camera_matrix(width, height), np.zeros((5, 1), dtype=np.float64)


def make_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params), dictionary
    return None, dictionary


def detect_marker(gray, detector, dictionary):
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=cv2.aruco.DetectorParameters()
        )
    if ids is None or len(ids) == 0:
        return None, None, None
    idx = 0
    if MARKER_ID is not None:
        matches = np.where(ids.flatten() == MARKER_ID)[0]
        if len(matches) == 0:
            return None, None, None
        idx = int(matches[0])
    return corners[idx], ids[idx], idx


def marker_object_points(side_length):
    s = side_length / 2.0
    return np.array(
        [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32
    )


def estimate_pose(corners, camera_matrix, dist_coeffs):
    image_points = corners[0] if corners.ndim == 3 else corners
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    obj_pts = marker_object_points(MARKER_LENGTH_M)
    flags = (
        cv2.SOLVEPNP_IPPE_SQUARE
        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
        else cv2.SOLVEPNP_ITERATIVE
    )
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, image_points, camera_matrix, dist_coeffs, flags=flags
    )
    if not ok:
        raise RuntimeError("solvePnP fallo")
    return rvec, tvec


def projection_from_k(K, width, height, znear, zfar):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 2.0 * fx / width
    P[1, 1] = 2.0 * fy / height
    P[0, 2] = (width  - 2.0 * cx) / width
    P[1, 2] = (2.0 * cy - height) / height
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -1.0
    P[3, 2] = -2.0 * zfar * znear / (zfar - znear)
    return P


def modelview_from_pose(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3]  = tvec.flatten()
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    return (cv_to_gl @ M).T.astype(np.float32)


_quadric    = None
_glut_ready = False


def init_glut_for_geometry():
    """
    Inicializa GLUT para la tetera.
    DEBE llamarse desde el hilo principal (loop de render),
    NUNCA desde un callback de teclado de GLFW.
    """
    global _glut_ready
    if _glut_ready:
        return
    from OpenGL.GLUT import glutInit
    glutInit(sys.argv if sys.argv else [""])
    _glut_ready = True


def draw_sphere(radius: float = 1.0) -> None:
    global _quadric
    if _quadric is None:
        _quadric = gluNewQuadric()
        gluQuadricDrawStyle(_quadric, GLU_FILL)
    gluSphere(_quadric, radius, 32, 16)


def draw_teapot(scale: float) -> None:
    from OpenGL.GLUT import glutSolidTeapot
    glutSolidTeapot(scale)


def draw_ar_object(mode: str, scale: float) -> None:
    glPushMatrix()
    glTranslatef(0.0, 0.0, scale * 0.5)
    if mode == "sphere":
        glColor3f(0.35, 0.75, 1.0)
        draw_sphere(scale)
    else:
        glColor3f(0.85, 0.45, 0.25)
        draw_teapot(scale)
    glPopMatrix()


def setup_lighting() -> None:
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightfv(GL_LIGHT0, GL_POSITION, (0.2, 0.4, 1.0, 0.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  (1.0, 1.0, 0.95, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT,  (0.25, 0.25, 0.25, 1.0))
    glEnable(GL_NORMALIZE)


_tex_id  = None
_tex_buf = None


def upload_frame_texture(frame_bgr, width, height) -> None:
    global _tex_id, _tex_buf
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.flip(rgb, 0)
    if _tex_buf is None or _tex_buf.shape[:2] != (height, width):
        _tex_buf = np.empty((height, width, 3), dtype=np.uint8)
    np.copyto(_tex_buf, rgb)
    if _tex_id is None:
        _tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, _tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
        GL_RGB, GL_UNSIGNED_BYTE, _tex_buf,
    )


def draw_background_quad(width, height) -> None:
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix(); glLoadIdentity()
    glOrtho(0, width, 0, height, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix(); glLoadIdentity()
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, _tex_id)
    glColor3f(1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(0, 0)
    glTexCoord2f(1, 0); glVertex2f(width, 0)
    glTexCoord2f(1, 1); glVertex2f(width, height)
    glTexCoord2f(0, 1); glVertex2f(0, height)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)


def draw_scene_3d(rvec, tvec, camera_matrix, width, height, mode, scale) -> None:
    P  = projection_from_k(camera_matrix, width, height, ZNear, ZFar)
    MV = modelview_from_pose(rvec, tvec)
    glMatrixMode(GL_PROJECTION); glLoadMatrixf(P)
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity(); glMultMatrixf(MV)
    setup_lighting()
    draw_ar_object(mode, scale)


# ===========================================================================
# SpringObject y HandTracker
# ===========================================================================

class SpringObject:
    def __init__(self):
        self.pos: np.ndarray             = np.zeros(3, dtype=np.float64)
        self.vel: np.ndarray             = np.zeros(3, dtype=np.float64)
        self.anchor: np.ndarray          = np.zeros(3, dtype=np.float64)
        self.anchor_rvec: np.ndarray | None = None
        self.has_anchor: bool            = False
        self.grabbed: bool               = False
        self.grab_offset: np.ndarray     = np.zeros(3, dtype=np.float64)

    def update_anchor(self, rvec, tvec):
        self.anchor      = tvec.flatten().copy()
        self.anchor_rvec = rvec.copy()
        self.has_anchor  = True
        if not self.grabbed:
            self.pos = self.anchor.copy()
            self.vel = np.zeros(3)

    def grab(self, hand_cam):
        self.grabbed     = True
        self.vel         = np.zeros(3)
        self.grab_offset = self.pos - hand_cam

    def release(self):
        self.grabbed = False

    def move_with_hand(self, hand_cam):
        if self.grabbed:
            self.pos = hand_cam + self.grab_offset

    def step(self, dt):
        if self.grabbed or not self.has_anchor:
            return
        diff = self.anchor - self.pos
        acc  = SPRING_K * diff - SPRING_D * self.vel
        self.vel += acc * dt
        self.pos += self.vel * dt

    @property
    def tvec_virtual(self):
        return self.pos.reshape(3, 1).astype(np.float64)


class HandTracker:
    def __init__(self):
        self.available = MEDIAPIPE_AVAILABLE
        if self.available:
            self.hands = _mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.55,
            )
        self.pinching: bool = False
        self.pinch_px: tuple[int, int] | None = None

    def process(self, frame_bgr):
        self.pinching = False
        self.pinch_px = None
        if not self.available:
            return
        h, w = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res  = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return
        lm = res.multi_hand_landmarks[0]
        _mp_drawing.draw_landmarks(frame_bgr, lm, _mp_hands.HAND_CONNECTIONS)
        thumb = lm.landmark[_mp_hands.HandLandmark.THUMB_TIP]
        index = lm.landmark[_mp_hands.HandLandmark.INDEX_FINGER_TIP]
        wrist = lm.landmark[_mp_hands.HandLandmark.WRIST]
        mid   = lm.landmark[_mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        palm_ref  = np.hypot(wrist.x - mid.x, wrist.y - mid.y) + 1e-6
        raw_dist  = np.hypot(thumb.x - index.x, thumb.y - index.y)
        norm_dist = raw_dist / palm_ref
        self.pinching = norm_dist < PINCH_THRESHOLD
        cx = int((thumb.x + index.x) / 2 * w)
        cy = int((thumb.y + index.y) / 2 * h)
        self.pinch_px = (cx, cy)
        color = (0, 255, 80) if self.pinching else (30, 160, 255)
        cv2.circle(frame_bgr, (cx, cy), 14, color, -1)
        cv2.circle(frame_bgr, (cx, cy), 14, (255, 255, 255), 2)
        label = "AGARRADO" if self.pinching else "mano"
        cv2.putText(frame_bgr, label, (cx + 18, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    def close(self):
        if self.available:
            self.hands.close()


def pinch_to_camera_coords(pinch_px, K, depth):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (pinch_px[0] - cx) / fx * depth
    y = (pinch_px[1] - cy) / fy * depth
    return np.array([x, y, depth], dtype=np.float64)


def project_to_screen(pt3d, rvec, tvec, K, dist):
    try:
        pts2d, _ = cv2.projectPoints(
            pt3d.reshape(1, 1, 3).astype(np.float32), rvec, tvec, K, dist)
        return int(pts2d[0, 0, 0]), int(pts2d[0, 0, 1])
    except Exception:
        return None


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    global OBJECT_MODE, MODEL_SCALE, _pending_glut_init

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("No se pudo abrir la camara.", file=sys.stderr)
        sys.exit(1)

    ret, probe = cap.read()
    if not ret:
        sys.exit(1)

    cam_h, cam_w = probe.shape[:2]
    camera_matrix, dist_coeffs = load_calibration(cam_w, cam_h)
    detector, dictionary = make_aruco_detector()

    # Inicializar GLUT en el hilo principal si el modo inicial es tetera
    if OBJECT_MODE == "teapot":
        init_glut_for_geometry()

    if not glfw.init():
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    window = glfw.create_window(cam_w, cam_h, WINDOW_TITLE, None, None)
    if not window:
        glfw.terminate()
        sys.exit(1)

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    def on_key(win, key, _scancode, action, _mods):
        """
        FIX #1: el callback ya NO llama a init_glut_for_geometry().
        Solo activa la bandera _pending_glut_init = True.
        La inicializacion real ocurre al inicio del siguiente frame,
        en el hilo principal donde OpenGL/GLUT tienen contexto valido.
        """
        global OBJECT_MODE, MODEL_SCALE, _pending_glut_init
        if action != glfw.PRESS:
            return
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_T:
            OBJECT_MODE = "sphere" if OBJECT_MODE == "teapot" else "teapot"
            if OBJECT_MODE == "teapot":
                _pending_glut_init = True   # <-- se procesa en el loop principal
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            MODEL_SCALE *= 1.1
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            MODEL_SCALE /= 1.1

    glfw.set_key_callback(window, on_key)
    glEnable(GL_DEPTH_TEST)

    spring_obj   = SpringObject()
    hand_tracker = HandTracker()
    prev_time    = glfw.get_time()
    was_pinching = False

    while not glfw.window_should_close(window):

        if _pending_glut_init:
            init_glut_for_geometry()
            _pending_glut_init = False

        ret, frame = cap.read()
        if not ret:
            continue

        now       = glfw.get_time()
        dt        = min(now - prev_time, 0.05)
        prev_time = now

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, _, _ = detect_marker(gray, detector, dictionary)
        if corners is not None:
            current_rvec, current_tvec = estimate_pose(
                corners, camera_matrix, dist_coeffs)
            spring_obj.update_anchor(current_rvec, current_tvec)

        hand_tracker.process(frame)
        pinching = hand_tracker.pinching
        pinch_px = hand_tracker.pinch_px

        obj_depth = float(spring_obj.pos[2]) if spring_obj.has_anchor else 0.40

        obj_px = None
        if spring_obj.anchor_rvec is not None and spring_obj.has_anchor:
            obj_px = project_to_screen(
                np.zeros(3, dtype=np.float32),
                spring_obj.anchor_rvec,
                spring_obj.tvec_virtual,
                camera_matrix, dist_coeffs,
            )

        if pinching and not was_pinching:
            if pinch_px is not None and obj_px is not None:
                dist_screen = np.hypot(
                    pinch_px[0] - obj_px[0],
                    pinch_px[1] - obj_px[1],
                )
                if dist_screen < GRAB_RADIUS_PX:    # ahora es 90, no 0
                    hand_cam = pinch_to_camera_coords(
                        pinch_px, camera_matrix, obj_depth)
                    spring_obj.grab(hand_cam)

        elif not pinching and was_pinching:
            if spring_obj.grabbed:
                spring_obj.release()

        was_pinching = pinching

        if spring_obj.grabbed and pinch_px is not None:
            hand_cam = pinch_to_camera_coords(
                pinch_px, camera_matrix, obj_depth)
            spring_obj.move_with_hand(hand_cam)

        spring_obj.step(dt)

        glViewport(0, 0, w, h)
        upload_frame_texture(frame, w, h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_background_quad(w, h)

        if spring_obj.has_anchor and spring_obj.anchor_rvec is not None:
            draw_scene_3d(
                spring_obj.anchor_rvec,
                spring_obj.tvec_virtual,
                camera_matrix, w, h,
                OBJECT_MODE, MODEL_SCALE,
            )

        glfw.swap_buffers(window)
        glfw.poll_events()

    hand_tracker.close()
    cap.release()
    glfw.terminate()


if __name__ == "__main__":
    main()