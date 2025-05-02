from OpenGL.GL import *
from OpenGL.GLUT import *

glutInit()
glutCreateWindow(b"Check GL Version".encode('utf-8'))
print("OpenGL version:", glGetString(GL_VERSION).decode())
print("GLSL version:", glGetString(GL_SHADING_LANGUAGE_VERSION).decode())