"""
流场环境，包含immersed boundary求解器，输入是流场的速度、压强、拉格朗日结构的
力、纤维结构的连接关系，输出是采取了某个变形策略后下一时间步的上述流场信息
"""
import numpy as np
import math
import pyfftw


class point: #存储该点所在的物理信息
     def __init__(self, x=0, y=0, u=0, v=0, p=0, fx=0, fy=0):
         self.x = x
         self.y = y
         self.u = u
         self.v = v
         self.p = p
         self.fx = fx
         self.fy = fy
         
class Flow_Field():
    def __init__(self):
        #每个时间步可以采取的策略是刚度增减5%，10%或不变
        self.action_space = ['plus5','plus10','minus5','minus10','zero']
        self.n_actions = len(self.action_space)
        #self.n_features = 2
        
        #存储整个流场
        
        #从input2d文件读入网格基本信息
        self.Lx,self.Ly,self.Nx,self.Ny,self.tStart,self.tFinal,self.dt=self.readGridInfo('input2d') 
        #流域为二维[0,Lx]x[0,Ly]
        #总网格为Nx x Ny
        #起始时间 tStart
        #终止时间 tFinal
        #离散时间步长 dt
        
        
        #AllPointInfo是一个列表，列表元素就是point
        AllPointInfo=self.readFieldinfo(self.tStart)
        
        
        #从起始时间所指定的拉格朗日点文件读入
        #存储初始时刻拉格朗日点的坐标[LagX,LagY]以及每个点上面的力[Fx,Fy]
        self.LagPoints=self.readVertexinfo('Vertex_'+str(self.tStart)+'.vtk') 
        self.Nb=len(self.LagPoints) #拉格朗日点的总数
        
        #存储每个拉格朗日点之间的连接关系[ID1 ID2 Kstiff],暂时让它不变
        self.spring=self.readSpring('Lagrange.spring') 
        
        #存储非恒定变形梁关系[ID1 ID2 ID3 Kstiff CX CY]
        self.non_Invarient_beam=self.readBeam('non_Inv_beam_'+str(self.tStart)+'.vtk') 
 
        
    #从input2d读入流场网格信息 
    def readGridInfo(self,file_name):
        grid=open(file_name,'r')
        info=[]
        for i in range(7):
            tmp=grid.readline()
            info.append(float(tmp))
        grid.close()
        return info[0],info[1],int(info[2]),int(info[3]),info[4],info[5],info[6]
        #return Lx,Ly,Nx,Ny,tStart,tFinal,dt
        
    #读入流场速度压强力等信息
    def readFieldinfo(self,tStart):
        field=[]
        file=open(str(tStart)+'.data')
        Ntotal=int(file.readline())
        for i in range(Ntotal):
            tmp=file.readline().split()
            element=point()
            element.x=float(tmp[0])
            element.y=float(tmp[1])
            element.u=float(tmp[2])
            element.v=float(tmp[3])
            element.p=float(tmp[4])
            element.fx=float(tmp[5])
            element.fy=float(tmp[6])
            field.append(element)
        file.close()
        return field
    
    #读入拉格朗日点和每个点上的力
    def readVertexinfo(self,file_name):
        vertex=[]
        file=open(file_name)
        Ntotal=int(file.readline())
        for i in range(Ntotal):
            tmp=file.readline().split()
            vertex.append([float(tmp[0]),float(tmp[1]),float(tmp[2]),float(tmp[3])])
        file.close()
        return vertex
    
    #读入spring
    def readSpring(self,file_name):
        spring=[]
        file=open(file_name,'r')
        Ntotal=int(file.readline())
        for i in range(Ntotal):
            tmp=file.readline().split()
            spring.append([int(tmp[0]),int(tmp[1]),float(tmp[2])])
        file.close()
        return spring
    
    #读入beam
    def readBeam(self,file_name):
        beam=[]
        file=open(file_name,'r')
        Ntotal=int(file.readline())
        for i in range(Ntotal):
            tmp=file.readline().split()
            beam.append([int(tmp[0]),int(tmp[1]),int(tmp[2]),float(tmp[3]),float(tmp[4]),float(tmp[5])])
        file.close()
        return beam
        
    
    
    #流体IB求解器,输入action后根据当前的流场信息求解下一步的流场并更新当前的流场状态
    #并返回s_:下一时刻的物体受力、质心坐标、最前端位置、对心力矩,reward:正相关于推进的距离
    #负相关于质心y方向偏离可以接受的范围的距离
    def step(self, action):
        
        
        return s_, reward, done




