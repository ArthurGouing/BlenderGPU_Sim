import taichi as ti
import numpy as np
import bpy

@ti.data_oriented
class Simulator():
    def __init__(self, **kwargs) -> None:
        # Architecture Computation
        if "arch" not in kwargs:
            self.arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
            kwargs.update({"arch": self.arch})
        ti.init(**kwargs)

        # Simulation Parameters 
        self.dt = 0.0002
        self.fps = 24
        self.substeps = int(1 / self.fps // self.dt) # Do fake automatated updated attribute
        self.curr_time =  0

        self.gravity = ti.Vector([0, 0, -9.81])
        self.spring_rigidity = 4e5
        self.spring_damping = 1e5
        self.drag_damping = 2 # 1.5

        self.bending_springs = True
        self.isnot_init = True


        # Print Simulation Informations
        print("")
        print("  Simulating with Mass spring method")
        print("")
        print("--Simulation parameters-----------------------")
        print(f" {'dt':<10}: {self.dt}")
        print(f" {'substeps':<10}: {self.substeps}")
        print("")
        print(f" {'gravity':<10}: {self.gravity}")
        print(f" {'rigidity':<10}: {self.spring_rigidity}")
        print(f" {'damping':<10}: {self.spring_damping}")
        print(f" {'drag':<10}: {self.drag_damping}")
        print("------------------------------------------------------")

        # self.ball_radius = 0.3
        # self.ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
        # self.ball_center[0] = [0, 0, 0]

    def initialize_from_obj(self, obj: bpy.types.Object):
        # Create Device fields
        self.n = len(obj.data.vertices)
        self.n_spring = len(obj.data.edges) # sum([len(e) for e in obj.data.edges], 0)
        self.num_triangles = len(obj.data.polygons) # (self.n - 1) * (self.n - 1) * 2
        
        # Fill positions from object mesh
        points = list()
        neighbor = list()
        springs = list()
        for i in range(self.n):
            neighbor.append(list())
        # Build points
        for v in obj.data.vertices:
            points.append([v.co.x, v.co.y, v.co.z])
        # Build edges
        for e in obj.data.edges:
            p_id1 = e.vertices[0]
            p_id2 = e.vertices[1]
            neighbor[p_id2].append(p_id1)
            neighbor[p_id1].append(p_id2)
        
        for i, neib in enumerate(neighbor):
            s = neib.copy()
            neighbor_of_neighbors = sum([neighbor[k] for k in neib], [])
            for neib_neib in neighbor_of_neighbors:
                if neib_neib not in s+[i]:
                    s += [neib_neib]
            springs.append(s)
        # Fill edges to undefined value
        self.max_spring = max( [len(springs_list) for springs_list in springs] )
        for e in springs:
            diff = self.max_spring - len(e)
            if diff > 0:
                e += diff*[-1] 
        
        self.points = np.array(points, dtype=np.single)
        self.neighbor = neighbor
        self.edges = np.array(springs, dtype=np.int32)

        # Init GPU fields
        self.create_fields()
        self.initialize_points()
        self.initialize_velocity()
        self.initialize_springs()

        del[points]
        del[neighbor]

        # Springs
        # self.initialize_springs()
        self.isnot_init = False
        print("")
        print("--Mesh parameters-----------------------")
        print(f" {'n':<10}: {self.n}")
        print(f" {'n spring':<10}: {self.n_spring}")
        print(f" {'max spring':<10}: {self.max_spring}")
        print(f" {'memory':<10}: {'TODO'}")
        print("------------------------------------------------------")
        # print("x")
        # print(self.x)
        # print("V")
        # print(self.v)
        # print("L0:")
        # print(self.l0)
        # print("edges")
        # print(self.edges)
        # print("springs:")
        # print(self.springs)

    def create_fields(self):
        self.x = ti.Vector.field(3, dtype=float, shape=self.n)
        self.v = ti.Vector.field(3, dtype=float, shape=self.n)

        # self.l0 = ti.field(float, shape=(self.n, self.max_spring))
        self.l0 = ti.Vector.field(self.max_spring, dtype=float, shape=self.n)
        self.springs = ti.Vector.field(self.max_spring, dtype=int, shape=self.n)

        # self.indices = ti.field(int, shape=self.num_triangles * 3)
        # self.vertices = ti.Vector.field(3, dtype=float, shape=self.n * self.n)
        # self.colors = ti.Vector.field(3, dtype=float, shape=self.n * self.n)

    def initialize_points(self):
        self.x.from_numpy(self.points)
    
    @ti.kernel
    def initialize_velocity(self):
        for i in self.x:
            self.v[i] = [0, 0, 0]

    def initialize_springs(self):
        # Init springs/edges size (before fields !!!)
        # Create spring ids list    # and springs from objects (neighbor list and poitns)
        self.springs.from_numpy(self.edges)

        # Create L0
        all_l0 = list()
        for i, neigh_list in enumerate(self.edges):
            xi = self.points[i]
            l0_list = list()
            for j in self.edges[i]:
                if j==-1: 
                    l0_list.append(0.0)
                    continue
                xj = self.points[j] # Write -1 if there is not j-th neighbor
                l0 = np.linalg.norm(xi-xj)
                l0_list.append(l0)
            all_l0.append(l0_list)
        self.l0.from_numpy(np.array(all_l0))

    def frame_forward(self):
        for t in range(self.substeps):
            self.step_forward()
            self.curr_time += t

    @ti.kernel
    def step_forward(self):
        for i in self.x:
            # self.x[i] += ti.Vector([0, 0, 0.001])
            x = self.x[i]
            v = self.v[i]
            l0 = self.l0[i]
            # Volumic Forces
            force = self.gravity

            # # Internal Forces
            for j in range(self.max_spring): # self.springs[i]:
                spring_id = self.springs[i][j]
                if spring_id==-1: 
                    continue
                spring_vec = self.x[spring_id] - x
                spring_vel = self.v[spring_id] - v
                dir = spring_vec.normalized()
                l = ti.math.length(spring_vec)

                f = self.spring_rigidity * (l-l0[j]) * dir
                f += spring_vel.dot(dir) * self.spring_damping * l*l
                force += self.spring_rigidity * (l-l0[j]) * dir
                # print("point:", i, "neighbor:", j, "xi=", x, "xj=", self.x[spring_id], "neighb_id", spring_id)
            v += force * self.dt
            v *= ti.exp(-self.drag_damping * self.dt)

            # # Surfacic Forces
            distance_to_sphere_center = x - ti.Vector([0.0, 0.0, 0.0])
            distance_to_sphere = distance_to_sphere_center.norm() - 1.0
            if distance_to_sphere<=0:
                # Velocity projection
                normal = distance_to_sphere_center.normalized()
                v -= ti.min(v.dot(normal), 0) * normal

            x += self.dt * v / 2

            # if i!=0 and i!=1:
            self.x[i] = x
            self.v[i] = v

    def update_vertices(self, obj: bpy.types.Object):
        import array
        vert = obj.data.vertices
        points_array = self.x.to_numpy().ravel().tolist()
        seq = array.array('f', points_array)
        vert.foreach_set('co', seq)
        print(self.x[2])
        obj.data.update()

    def reset(self):
        self.x.from_numpy(self.points)
        self.initialize_velocity()
