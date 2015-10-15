#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Bump mapping is a way of making polygonal surfaces look
# less flat.  This sample uses normal mapping for all
# surfaces, and also parallax mapping for the column.
#
# This is a tutorial to show how to do normal mapping
# in panda3d using the Shader Generator.

import scipy.misc
import sys
import os
import numpy as np
import random

from panda3d.core import loadPrcFileData
from direct.showbase.ShowBase import ShowBase
from panda3d.core import CollisionTraverser, CollisionNode
from panda3d.core import CollisionSphere, CollisionPlane, CollisionTube
from panda3d.core import CollisionBox
from panda3d.core import Plane, Vec3, Point3
from panda3d.core import CollideMask
from panda3d.core import CollisionHandlerQueue, CollisionRay
from panda3d.core import WindowProperties
from panda3d.core import Filename, Shader
from panda3d.core import AmbientLight, PointLight
from panda3d.core import PandaNode, NodePath, Camera, TextNode
from panda3d.core import LPoint3, LVector3
from direct.task.Task import Task
from direct.actor.Actor import Actor
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.DirectObject import DirectObject
from direct.filter.CommonFilters import *


class Demo(ShowBase):
    _speed = 25

    def reset(self):
        x = random.random() * 40 - 20
        y = random.random() * 40 - 20
        self.ralph.setPos((x, y, 0))

    def __init__(self, img_size=512, screen_off=True):
        self.img_size = img_size
        loadPrcFileData("", "transform-cache false")
        loadPrcFileData("", "audio-library-name null")  # Prevent ALSA errors
        loadPrcFileData("", "win-size %d %d" % (img_size, img_size))
        loadPrcFileData("", "parallax-mapping-samples 3\n"
                            "parallax-mapping-scale 0.1")

        if screen_off:
            # Spawn an offscreen buffer
            loadPrcFileData("", "window-type offscreen")
        # Initialize the ShowBase class from which we inherit, which will
        # create a window and set up everything we need for rendering into it.
        ShowBase.__init__(self)

        # Load the 'abstract room' model.  This is a model of an
        # empty room containing a pillar, a pyramid, and a bunch
        # of exaggeratedly bumpy textures.

        self.room = loader.loadModel("models/abstractroom")
        self.room.reparentTo(render)

        # Create the main character, Ralph

        self.ralph = Actor("models/ralph",
                           {"run": "models/ralph-run",
                            "walk": "models/ralph-walk"})
        self.ralph.reparentTo(render)
        self.ralph.setScale(.2)
        self.reset()

        self.pieces = [Piece(self.room) for _ in range(200)]
        ##################################################
        cnodePath = self.room.attachNewNode(CollisionNode('cnode'))
        plane = CollisionPlane(Plane(Vec3(1, 0, 0), Point3(-60, 0, 0)))  # left
        cnodePath.node().addSolid(plane)
        plane = CollisionPlane(
            Plane(Vec3(-1, 0, 0), Point3(60, 0, 0)))  # right
        cnodePath.node().addSolid(plane)
        plane = CollisionPlane(Plane(Vec3(0, 1, 0), Point3(0, -60, 0)))  # back
        cnodePath.node().addSolid(plane)
        plane = CollisionPlane(
            Plane(Vec3(0, -1,  0), Point3(0, 60,  0)))  # front
        cnodePath.node().addSolid(plane)

        sphere = CollisionSphere(-25, -25, 0, 12.5)
        cnodePath.node().addSolid(sphere)

        box = CollisionBox(Point3(5, 5, 0), Point3(45, 45, 10))
        cnodePath.node().addSolid(box)

        # Make the mouse invisible, turn off normal mouse controls
        self.disableMouse()
        self.camLens.setFov(80)

        # Set the current viewing target
        self.focus = LVector3(55, -55, 20)
        self.heading = 180
        self.pitch = 0
        self.mousex = 0
        self.mousey = 0
        self.last = 0
        self.mousebtn = [0, 0, 0]

        # Add a light to the scene.
        self.lightpivot = render.attachNewNode("lightpivot")
        self.lightpivot.setPos(0, 0, 25)
        self.lightpivot.hprInterval(10, LPoint3(360, 0, 0)).loop()
        plight = PointLight('plight')
        plight.setColor((5, 5, 5, 1))
        plight.setAttenuation(LVector3(0.7, 0.05, 0))
        plnp = self.lightpivot.attachNewNode(plight)
        plnp.setPos(45, 0, 0)
        self.room.setLight(plnp)

        # Add an ambient light
        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = render.attachNewNode(alight)
        self.room.setLight(alnp)

        # Create a sphere to denote the light
        sphere = loader.loadModel("models/icosphere")
        sphere.reparentTo(plnp)

        self.cameraModel = self.ralph
        camera.reparentTo(self.cameraModel)
        camera.setZ(2)

        self.cTrav = CollisionTraverser()

        cs = CollisionSphere(0, 0, 0, 1)
        cnodePath = self.ralph.attachNewNode(CollisionNode('cnode'))
        cnodePath.node().addSolid(cs)

        cnodePath.show()
        self.ralphGroundHandler = CollisionHandlerQueue()
        self.cTrav.addCollider(cnodePath, self.ralphGroundHandler)

        self.cnodePath = cnodePath

        # Tell Panda that it should generate shaders performing per-pixel
        # lighting for the room.
        self.room.setShaderAuto()

        self.shaderenable = 1

        tex = Texture()
        self.depthmap = tex
        tex.setFormat(Texture.FDepthComponent)
        altBuffer = self.win.makeTextureBuffer(
            "hello", img_size, img_size, tex, True)
        self.altBuffer = altBuffer
        altCam = self.makeCamera(altBuffer)
        altCam.reparentTo(self.ralph)  # altRender)
        altCam.setZ(2)
        l = altCam.node().getLens()
        l.setFov(80)
        ################### end init

    def step(self, rotation):
        dt = 0.02

        self.ralph.setH(self.ralph.getH() + rotation * dt)
        self.ralph.setY(self.ralph, self._speed * dt)
        for _ in range(5):
            random.choice(self.pieces).step(dt)

        f = taskMgr.getAllTasks()[-1].getFunction()
        f(None)
        f = taskMgr.getAllTasks()[2].getFunction()
        f(None)
        entries = (self.ralphGroundHandler.getEntries())
        if len(entries) > 0:
            self.reset()
            return 1
        else:
            return 0

    def resetGame(self):
        while True:
            self.reset()
            for p in self.pieces:
                p.reset()
            if 0 == self.step(0):
                break

    def renderFrame(self):
        base.graphicsEngine.renderFrame()

    def getDepthMapT(self):
        tex = self.depthmap
        #tex = self.altBuffer.getTexture()
        r = tex.getRamImage()
        s = r.getData()
        i = np.fromstring(s, dtype='float32')
        i = i.reshape((self.img_size, self.img_size))
        i = np.flipud(i)
        return i


class Piece:

    def __init__(self, room, roomsize=60, speed_max=1, acceleration_max=1):
        self.roomsize = roomsize
        self.speed_max = speed_max
        self.acceleration_max = acceleration_max

        p = 'models/knight',    "models/pawn",    "models/king",    "models/queen",    "models/bishop",    "models/knight",    "models/rook"
        p = random.choice(p)
        self.knight = loader.loadModel(p)

        color = random.random(), random.random(), random.random(), 1
        tex = loader.loadTexture('./models/wood.jpg')
        self.knight.setTexture(tex)

        self.knight.reparentTo(room)
        self.model = self.knight

        cs = CollisionSphere(0, 0, 0, 0.27)
        cnodePath = self.knight.attachNewNode(CollisionNode('cnode'))
        cnodePath.node().addSolid(cs)

        self.reset()

    def reset(self):
        self._speed = np.zeros(2)
        x = random.random() * 2 * self.roomsize - self.roomsize
        y = random.random() * 2 * self.roomsize - self.roomsize
        self._pos = np.asarray([x, y])
        k_pos = x, y, 0
        self.knight.setPos(k_pos)
        piece_scale = 9 * random.random() + 2
        self.knight.setScale(piece_scale)

    def step(self, dt):
        dx = self.acceleration_max * (2 * rand() - 1)
        dy = self.acceleration_max * (2 * rand() - 1)
        self._speed += dx * dt, dy * dt
        norm = np.linalg.norm(self._speed)
        if norm > self.speed_max:
            self._speed *= self.speed_max / norm
        self._pos += self._speed * dt
        if self._pos[0] > self.roomsize:
            self._pos[0] = self.roomsize
            self._speed[0] = -abs(self._speed[0])
        if self._pos[0] < -self.roomsize:
            self._pos[0] = -self.roomsize
            self._speed[0] = abs(self._speed[0])
        if self._pos[1] > self.roomsize:
            self._pos[1] = self.roomsize
            self._speed[1] = -abs(self._speed[1])
        if self._pos[1] < -self.roomsize:
            self._pos[1] = -self.roomsize
            self._speed[1] = abs(self._speed[1])
        self.model.setPos((self._pos[0], self._pos[1], 0))


def rand():
    return random.random()
