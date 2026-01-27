import numpy as np
import xml.etree.ElementTree as ET


def angles_321_dcm(rpy):
    """
    Equivalent to the MATLAB Angles321_DCM(rpy'),
    which applies rotations in Z-Y-X order (3-2-1).
    rpy: [roll, pitch, yaw] in radians
         (roll about X, pitch about Y, yaw about Z)
    """
    rx, ry, rz = rpy

    # Rotation about z (yaw)
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,           0,          1]
    ])

    # Rotation about y (pitch)
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0,          1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # Rotation about x (roll)
    Rx = np.array([
        [1, 0,          0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)]
    ])

    # Combined rotation for 3-2-1 (Z-Y-X)
    return Rz @ Ry @ Rx


def make_transform(xyz=None, rpy=None):
    """
    Create a 4x4 homogeneous transform from xyz translation and RPY rotation.
    xyz, rpy: length-3 lists/tuples (in meters, radians)
    """
    if xyz is None:
        xyz = [0.0, 0.0, 0.0]
    if rpy is None:
        rpy = [0.0, 0.0, 0.0]

    T = np.eye(4)
    T[0:3, 0:3] = angles_321_dcm(rpy)
    T[0:3, 3] = xyz
    return T


def transform_inv(T):
    """
    Invert a 4x4 homogeneous transform.
    """
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    T_inv = np.eye(4)
    T_inv[0:3, 0:3] = R.T
    T_inv[0:3, 3] = -R.T @ p
    return T_inv


def connectivity_map(robot):
    n = robot['n_links_joints']
    if n == 0:
        return None, None, None

    branch = np.zeros((n, n), dtype=int)
    child = np.zeros((n, n), dtype=int)
    child_base = np.zeros(n, dtype=int)

    # Populate branch
    for i in reversed(range(n)):
        branch[i, i] = 1
        last_parent = i
        while True:
            pj_id = robot['links'][last_parent]['parent_joint']
            if pj_id == 0:
                break
            p_link_id = robot['joints'][pj_id - 1]['parent_link']
            if p_link_id == 0:
                break
            p_link_idx = p_link_id - 1
            branch[i, p_link_idx] = 1
            last_parent = p_link_idx

    # Populate child & child_base
    for i in reversed(range(n)):
        pj_id = robot['links'][i]['parent_joint']
        if pj_id == 0:
            child_base[i] = 1
        else:
            p_link_id = robot['joints'][pj_id - 1]['parent_link']
            if p_link_id == 0:
                child_base[i] = 1
            else:
                p_link_idx = p_link_id - 1
                child[i, p_link_idx] = 1

    return branch, child, child_base


def urdf2robot(filename, verbose_flag=False):
    """
    Reads a URDF file and returns a dictionary-based robot model,
    similar in spirit to the MATLAB urdf2robot function.
    """
    # Parse URDF
    tree = ET.parse(filename)
    root = tree.getroot()

    if root.tag != 'robot':
        raise ValueError("Root of URDF is not <robot>.")

    robot_name = root.attrib.get('name', 'unnamed_robot')

    # Collect top-level <link> and <joint> elements
    link_xml_list = list(root.findall('link'))
    joint_xml_list = list(root.findall('joint'))
    # NOTE: If your XML library doesn't have getparent, you can skip or handle differently
    # but for standard usage in python, you might do direct children only.

    # If we can't rely on getparent in your environment, do:
    # link_xml_list = root.findall('link')
    # joint_xml_list = root.findall('joint')

    if verbose_flag:
        print(f"Robot name: {robot_name}")
        print(f"Number of links (including base): {len(link_xml_list)}")

    # Initialize the robot dictionary
    robot = {
        'name': robot_name,
        'n_q': 0,
        'n_links_joints': len(link_xml_list) - 1,  # minus base
        'base_link': {},
        'links': [],
        'joints': []
    }

    # We store link data in a dict: links_map[link_name] = link_info
    links_map = {}
    for link_xml in link_xml_list:
        link_name = link_xml.attrib.get('name', '')
        link_info = {
            'name': link_name,
            'T': np.eye(4),  # store inertial offset transform
            'parent_joint': [],
            'child_joint': [],
            'mass': 0.0,
            'inertia': np.zeros((3, 3))
        }
        # <inertial>
        inertial = link_xml.find('inertial')
        if inertial is not None:
            # <origin xyz="" rpy="">
            origin = inertial.find('origin')
            if origin is not None:
                xyz_str = origin.attrib.get('xyz', '')
                rpy_str = origin.attrib.get('rpy', '')
                if xyz_str:
                    xyz = [float(x) for x in xyz_str.split()]
                else:
                    xyz = [0.0, 0.0, 0.0]
                if rpy_str:
                    rpy = [float(x) for x in rpy_str.split()]
                else:
                    rpy = [0.0, 0.0, 0.0]
                link_info['T'] = make_transform(xyz, rpy)

            # <mass value="">
            mass_el = inertial.find('mass')
            if mass_el is not None:
                mass_val = mass_el.attrib.get('value', '0')
                link_info['mass'] = float(mass_val)

            # <inertia ixx="" iyy="" izz="" ixy="" iyz="" ixz="">
            inertia_el = inertial.find('inertia')
            if inertia_el is not None:
                ixx = float(inertia_el.attrib.get('ixx', 0))
                iyy = float(inertia_el.attrib.get('iyy', 0))
                izz = float(inertia_el.attrib.get('izz', 0))
                ixy = float(inertia_el.attrib.get('ixy', 0))
                iyz = float(inertia_el.attrib.get('iyz', 0))
                ixz = float(inertia_el.attrib.get('ixz', 0))
                link_info['inertia'] = np.array([
                    [ixx, ixy, ixz],
                    [ixy, iyy, iyz],
                    [ixz, iyz, izz]
                ])

        links_map[link_name] = link_info

    # Similarly, parse joints
    joints_map = {}
    for joint_xml in joint_xml_list:
        joint_name = joint_xml.attrib.get('name', '')
        joint_type_name = joint_xml.attrib.get('type', 'fixed')
        # default to 'fixed' if not specified

        # Create joint dict
        joint_info = {
            'name': joint_name,
            'type_name': joint_type_name,
            'type': 0,  # 0 = fixed, 1 = revolute/continuous, 2 = prismatic
            'parent_link': '',
            'child_link': '',
            'T': np.eye(4),  # parent link frame
            'axis': np.array([0.0, 0.0, 0.0])
        }
        if joint_type_name in ['revolute', 'continuous']:
            joint_info['type'] = 1
        elif joint_type_name == 'prismatic':
            joint_info['type'] = 2
        elif joint_type_name == 'fixed':
            joint_info['type'] = 0
        else:
            raise ValueError(f"Unsupported joint type: {joint_type_name}")

        # <origin>
        origin = joint_xml.find('origin')
        if origin is not None:
            xyz_str = origin.attrib.get('xyz', '')
            rpy_str = origin.attrib.get('rpy', '')
            if xyz_str:
                xyz = [float(x) for x in xyz_str.split()]
            else:
                xyz = [0.0, 0.0, 0.0]
            if rpy_str:
                rpy = [float(x) for x in rpy_str.split()]
            else:
                rpy = [0.0, 0.0, 0.0]
            joint_info['T'] = make_transform(xyz, rpy)

        # <axis>
        axis_el = joint_xml.find('axis')
        if axis_el is not None:
            axis_str = axis_el.attrib.get('xyz', '')
            if axis_str:
                axis_vals = [float(x) for x in axis_str.split()]
                joint_info['axis'] = np.array(axis_vals)
            elif joint_info['type'] != 0:
                # A moving joint must have an axis
                raise ValueError(f"Joint {joint_name} is moving but has no axis.")

        # <limit>
        limit_el = joint_xml.find('limit')
        joint_info['limit'] = {'lower': -1e9, 'upper': 1e9} # Default to large range
        if limit_el is not None:
            lower_str = limit_el.attrib.get('lower')
            upper_str = limit_el.attrib.get('upper')
            if lower_str:
                joint_info['limit']['lower'] = float(lower_str)
            if upper_str:
                joint_info['limit']['upper'] = float(upper_str)

        # <parent>
        parent_el = joint_xml.find('parent')
        if parent_el is not None:
            parent_name = parent_el.attrib.get('link', '')
            joint_info['parent_link'] = parent_name
            # Register this joint in the parent's child_joint
            if parent_name in links_map:
                links_map[parent_name]['child_joint'].append(joint_name)

        # <child>
        child_el = joint_xml.find('child')
        if child_el is not None:
            child_name = child_el.attrib.get('link', '')
            joint_info['child_link'] = child_name
            # Register this joint in the child's parent_joint
            if child_name in links_map:
                links_map[child_name]['parent_joint'].append(joint_name)
        joint_info['T'] = transform_inv(links_map[joint_info['parent_link']]['T']) @ joint_info['T']
        joints_map[joint_name] = joint_info

    # Identify base link (the link with no parent joint)
    base_link_name = None
    for ln, linfo in links_map.items():
        if len(linfo['parent_joint']) == 0:
            base_link_name = ln
            break
    if base_link_name is None:
        raise ValueError("No link found with no parent_joint => no single base link?")

    if verbose_flag:
        print(f"Base link: {base_link_name}")

    # Fill robot['base_link']
    base_link_info = links_map[base_link_name]
    robot['base_link']['mass'] = base_link_info['mass']
    robot['base_link']['inertia'] = base_link_info['inertia']

    # ID maps
    robot_keys = {
        'link_id': {},
        'joint_id': {},
        'q_id': {}
    }

    # Base link is ID=0
    robot_keys['link_id'][base_link_name] = 0

    # We will store final link/joint data in robot['links'] / robot['joints']
    robot['links'] = []
    robot['joints'] = []

    # Recursively traverse
    def recurse_through_tree(parent_link_name, child_joint_name, link_id, joint_id, q_id):
        """
        Recursively fill in robot['links'] and robot['joints'] with unique IDs.
        link_id, joint_id: current counters.
        q_id: next joint variable index (like MATLAB's n_q).
        """
        jinfo = joints_map[child_joint_name]
        parent_id = robot_keys['link_id'][parent_link_name]
        child_link_name = jinfo['child_link']

        # Create a new joint entry
        new_joint_id = joint_id + 1
        new_joint = {
            'name': jinfo['name'],
            'id': new_joint_id,
            'type': jinfo['type'],
            'q_id': -1,  # default if fixed
            'parent_link': parent_id,
            'child_link': link_id + 1,  # new link will get assigned
            'axis': jinfo['axis'],
            'limit': jinfo['limit'],
            'T': jinfo['T']
        }
        # If it's revolute or prismatic, assign a q_id
        if jinfo['type'] in [1, 2]:
            new_joint['q_id'] = q_id
            robot_keys['q_id'][child_joint_name] = q_id
            q_id += 1

        robot['joints'].append(new_joint)
        robot_keys['joint_id'][child_joint_name] = new_joint_id

        # Create a new link entry
        new_link_id = link_id + 1
        cinfo = links_map[child_link_name]
        robot['links'].append({
            'name': cinfo['name'],
            'id': new_link_id,
            'parent_joint': new_joint_id,
            'T': cinfo['T'],
            'mass': cinfo['mass'],
            'inertia': cinfo['inertia']
        })
        robot_keys['link_id'][child_link_name] = new_link_id

        # Recurse for that child link's child joints
        for cjname in cinfo['child_joint']:
            robot_, keys_, lid, jid, qid = recurse_through_tree(
                child_link_name, cjname, new_link_id, new_joint_id, q_id
            )
            # update counters
            q_id = qid
            new_joint_id = jid
            new_link_id = lid

        return robot, robot_keys, new_link_id, new_joint_id, q_id

    # Start recursion from the base link’s child joints
    base_children = base_link_info['child_joint']

    # Initialize counters matching MATLAB style
    nl = 0   # base link was assigned ID=0 already
    nj = 0   # no joints so far
    nq = 1   # next joint var ID starts at 1

    for child_jn in base_children:
        # pass in current counters, get updated
        robot, robot_keys, nl, nj, nq = recurse_through_tree(
            base_link_name, child_jn, nl, nj, nq
        )

    # robot['n_q'] => total joint variables
    robot['n_q'] = nq - 1
    if verbose_flag:
        print(f"Number of joint variables: {robot['n_q']}")

    # Add connectivity map if needed
    branch, child, child_base = connectivity_map(robot)
    robot['con'] = {
        'branch': branch,
        'child': child,
        'child_base': child_base
    }

    return robot, robot_keys


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python urdf2robot.py <your_urdf_file>")
        sys.exit(1)

    robot_model, robot_keys = urdf2robot(sys.argv[1], verbose_flag=True)
    print("\n=== Robot Model ===")
    print("name:", robot_model['name'])
    print("n_q:", robot_model['n_q'])
    print("n_links_joints (excluding base):", robot_model['n_links_joints'])
    print("base_link:", robot_model['base_link'])
    print("\nlinks:", robot_model['links'])
    print("\njoints:", robot_model['joints'])
    print("\nkeys:", robot_keys)
    print("\n=== Connectivity Map ===")
    print(robot_model['con'])
