"""MuJoCo 기반 self-collision 체크.

Usage:
    checker = CollisionChecker("assets/spacerobot_collision.xml")
    has_collision = checker.check(q_joints)           # single config
    mask = checker.check_batch(q_joints_batch)        # batch
"""

import numpy as np
import mujoco


class CollisionChecker:
    """MuJoCo model로 self-collision 판정."""

    def __init__(self, mjcf_path: str):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

    def _set_config(self, q_joints: np.ndarray):
        """관절 위치 세팅 후 forward kinematics 계산."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[3] = 1.0          # base quaternion w=1 (identity)
        self.data.qpos[7:13] = q_joints   # 6 joints
        mujoco.mj_forward(self.model, self.data)

    def check(self, q_joints: np.ndarray) -> bool:
        """단일 관절 배열 [6] → self-collision 여부 (True=충돌)."""
        self._set_config(q_joints)
        return self.data.ncon > 0

    def check_batch(self, q_batch: np.ndarray) -> np.ndarray:
        """배치 [N, 6] → bool 배열 [N] (True=충돌)."""
        result = np.empty(len(q_batch), dtype=bool)
        for i, q in enumerate(q_batch):
            result[i] = self.check(q)
        return result

    def get_contacts(self, q_joints: np.ndarray) -> list[dict]:
        """충돌 상세 정보 반환 (디버깅용)."""
        self._set_config(q_joints)
        contacts = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            contacts.append({
                "geom1": geom1 or f"geom_{c.geom1}",
                "geom2": geom2 or f"geom_{c.geom2}",
                "dist": c.dist,
                "pos": c.pos.copy(),
            })
        return contacts
