import math
import os
import time
from calendar import c

import pybullet as p
import pybullet_data as pd

# GUIモードでPybulletに接続
p.connect(p.GUI)

# Pybulletのデータまでのパスを設定
p.setAdditionalSearchPath(pd.getDataPath())

# テクスチャIDを無効値で初期化
textureId = -1

# 地形生成をプログラムにより生成することを示す定数
useProgrammatic = 0

# 地形生成をPNGファイルにより生成することを示す定数
useTerrainFromPNG = 1

# 地形生成をCSVファイルから生成することを示す定数（実際は、なぜかtxtファイルを読み込んでいる）
useDeepLocoCSV = 2

# 地形の更新が必要かどうかを示す変数
updateHeightfield = False

# 地形生成方法をプログラムにより生成する方法に設定
heightfieldSource = useProgrammatic

import random

# 乱数のシードを設定
random.seed(10)

# デバッグビジュアライザを無効化
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

# 地形の変動範囲
heightPerturbationRange = 0.00

# プログラムによる地形生成を行う
if heightfieldSource == useProgrammatic:
    numHeightfieldRows = 256
    numHeightfieldColumns = 256
    heightfieldData = [0] * numHeightfieldRows * numHeightfieldColumns
    for j in range(int(numHeightfieldColumns / 2)):
        for i in range(int(numHeightfieldRows / 2)):
            height = random.uniform(0, heightPerturbationRange)
            heightfieldData[2 * i + 2 * j * numHeightfieldRows] = height
            heightfieldData[2 * i + 1 + 2 * j * numHeightfieldRows] = height
            heightfieldData[2 * i + (2 * j + 1) * numHeightfieldRows] = height
            heightfieldData[2 * i + 1 + (2 * j + 1) * numHeightfieldRows] = height

    # 地形の衝突形状を作成し、位置と向きを設定
    terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=[0.1, 0.1, 1], heightfieldTextureScaling=numHeightfieldRows, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns)
    terrain = p.createMultiBody(0, terrainShape)
    p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])

# 地形の色情報を変更
p.changeVisualShape(terrain, -1, rgbaColor=[1, 1, 1, 1])

# 球と箱の衝突形状を設定
sphereRadius = 0.05
colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sphereRadius, sphereRadius, sphereRadius])

# リンクの質量、衝突形状、位置、向き、関節パラメータを設定
mass = 1
visualShapeId = -1
link_Masses = [1]
linkCollisionShapeIndices = [colBoxId]
linkVisualShapeIndices = [-1]
linkPositions = [[0, 0, 0.11]]
linkOrientations = [[0, 0, 0, 1]]
linkInertialFramePositions = [[0, 0, 0]]
linkInertialFrameOrientations = [[0, 0, 0, 1]]
indices = [0]
jointTypes = [p.JOINT_REVOLUTE]
axis = [[0, 0, 1]]

# デバッグビジュアライザのレンダリングを有効化
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

# 重力、タイムステップを定義
GRAVITY = -9.8
dt = 1e-3

# 重力を設定
p.setGravity(0, 0, GRAVITY)

# タイムステップを設定
p.setTimeStep(dt)

# 二足歩行ロボットのurdfファイルを読み込む
cubeStartPos = [0, 0, 1.13]
cubeStartOrientation = p.getQuaternionFromEuler([0.0, 0, 0])
botId = p.loadURDF("/Users/itomasaki/Desktop/PixyzRL/examples/mybot.urdf", cubeStartPos, cubeStartOrientation)

# 二足歩行ロボットの各ジョイントを位置制御モードに設定
jointFrictionForce = 1
for joint in range(p.getNumJoints(botId)):
    p.setJointMotorControl2(botId, joint, p.POSITION_CONTROL, force=jointFrictionForce)

# リアルタイムシミュレーションを有効化
p.setRealTimeSimulation(1)


def collision_check(source, target, source_link_index=-1, target_link_index=-1):
    # 衝突情報を取得
    if source_link_index == -1 and target_link_index == -1:
        contact_points = p.getContactPoints(source, target)
    else:
        contact_points = p.getContactPoints(source, target, linkIndexA=source_link_index, linkIndexB=target_link_index)

    if contact_points is None:
        return False

    # 衝突情報があるかどうかを返す
    return len(contact_points) > 0


def reset():
    # ロボットの位置をリセット
    p.resetBasePositionAndOrientation(botId, [0, 0, 1.13], [0, 0, 0, 1])
    # ロボットのジョイントをリセット
    for joint_idx in range(p.getNumJoints(botId)):
        p.resetJointState(botId, joint_idx, 0)

    # 地形の位置をリセット
    p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])


# ループを繰り返す
while 1:
    p.stepSimulation()
    for joint_idx in range(p.getNumJoints(botId)):
        p.setJointMotorControl2(botId, joint_idx, p.POSITION_CONTROL, targetPosition=0)
    time.sleep(dt)
    body_height = p.getLinkState(botId, 0)[0][2]
    print(body_height)

    # ロボットが地形と衝突しているかどうかを確認
    print(collision_check(botId, terrain))

    # ロボットのbase_linkが地形と衝突しているかどうかを確認
    print(collision_check(botId, terrain, source_link_index=0))

    if collision_check(botId, terrain, source_link_index=0):
        reset()
