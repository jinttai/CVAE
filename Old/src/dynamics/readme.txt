### 로봇 파라미터 딕셔너리 ### 

robot = {
    'name': 'robot_name',  # 로봇의 이름
    'n_q': 0,  # 조인트 변수 개수
    'n_links_joints': 0,  # 링크와 조인트 개수 (베이스 제외)
    'base_link': {
        'mass': 0.0,  # 베이스 링크 질량
        'inertia': np.zeros((3, 3))  # 베이스 링크 관성 행렬
    },
    'links': [  # 링크 리스트 (ID 기반)
        {
            'id': 0,  # 링크 ID
            'parent_joint': 0,  # 부모 조인트 ID
            'T': np.eye(4),  # 링크 변환 행렬 (4x4)
            'mass': 0.0,  # 링크 질량
            'inertia': np.zeros((3, 3))  # 링크 관성 행렬
        },
        # 다른 링크들이 추가됨...
    ],
    'joints': [  # 조인트 리스트 (ID 기반)
        {
            'id': 0,  # 조인트 ID
            'type': 0,  # 조인트 타입 (0=고정, 1=회전, 2=프리스매틱)
            'q_id': -1,  # 조인트 변수 ID (-1이면 없음)
            'parent_link': 0,  # 부모 링크 ID
            'child_link': 1,  # 자식 링크 ID
            'axis': np.array([0.0, 0.0, 0.0]),  # 조인트 축
            'T': np.eye(4)  # 조인트 변환 행렬 (4x4)
        },
        # 다른 조인트들이 추가됨...
    ],
    'con': {  # 연결 관계
        'branch': np.zeros((0, 0), dtype=int),  # 브랜치 연결 행렬
        'child': np.zeros((0, 0), dtype=int),  # 자식 연결 행렬
        'child_base': np.zeros(0, dtype=int)  # 베이스 링크에 연결된 자식 링크
    }
}



### parameters
# t0: 6x1 vector, twist of the base frame projected in the inertial CCS
# tL: 6xn matrix, twists of the links projected in the inertial CCS
# P0: 6x6 matrix, twist propagation vector of the base frame
# pm: 6xn matrix, manipulator twist propagation vector
# Bi0: 6x6xn matrix, twist propagation matrix of the base frame
# Bij: 6x6xnxn matrix, twist propagation matrix of the links
# u0: 6x1 vector, Base-link velocities [\omega,rdot]. The angular velocity is projected in the body-fixed CCS, while the linear velocity is projected in the inertial CCS
# um: nx1 vector, Joint velocities
# u0dot: 6x1 vector, Base-link accelerations [\dot{\omega},\ddot{r}]. The angular acceleration is projected in the body-fixed CCS, while the linear acceleration is projected in the inertial CCS
# umdot: nx1 vector, Joint accelerations
# robot: dictionary, robot model
# t0dot: 6x1 vector, time derivative of the twist of the base frame projected in the inertial CCS
# tLdot: 6xn matrix, time derivative of the twists of the links projected in the inertial CCS

