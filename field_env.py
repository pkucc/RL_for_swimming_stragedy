"""
流场环境，调用matlab由当前时间步求解下一个时间步
由下一时间步的数据计算得出下一时间步物体的受力、质心、对心力矩、第一个点的x坐标
一个状态state包含了质心、物体受力、对心力矩 [s[0],s[1]]是质心，[s[2],s[3]]是总受力，s[4]是对心力矩
"""

import matlab.engine
import os
import math

class Flow_Field():
    def __init__(self):
        #每个时间步可以采取的策略是刚度增减5%，10%或不变
        self.action_space = ['plus5', 'plus10', 'minus5', 'minus10', 'zero']
        self.n_actions = len(self.action_space)
        self.n_features = 5
        self.episode = 0  # 当前训练次数
        self.startTime = 0  # 模拟起始时间
        self.endTime = self.read_end_time()  # 终止时间
        self.dt = self.read_dt()  # 时间间隔
        # currentTime=dt*currentTimeStep
        self.currenTimeStep = 0  # 当前时间步，第0步为t=0
        self.currentTime = 0  # 当前时间
        self.Lagpoints, self.fx, self.fy = self.read_LagForce(self.currenTimeStep)  # 读入当前时间步的拉格朗日点和力
        self.massCenter = self.calculate_mass_center(self.Lagpoints)  # 计算质心坐标
        self.torque = self.calculate_torque(self.Lagpoints, self.massCenter, self.fx, self.fy)  # 计算对质心的力矩
        self.Force = self.calculate_force(self.Lagpoints, self.fx, self.fy)  # 计算总受力
        self.eng = matlab.engine.start_matlab() ############
        self.k_beam = 0
        self.total_reward = 0

    # 从输入文件读入终止时间
    def read_end_time(self):
        pointer = open('input2d', 'r')
        lines = pointer.readlines()
        pointer.close()
        for line in lines:
            tmp = line.split()
            if len(tmp) == 0 or tmp[0] != 'Tfinal':
                continue
            else:
                return float(tmp[2])

    # 读入dt
    def read_dt(self):
        pointer = open('input2d', 'r')
        lines = pointer.readlines()
        pointer.close()
        #dump, dt = 0, 0
        for line in lines:
            tmp = line.split()
            if len(tmp) > 0 and tmp[0] == 'dt':
                dt = float(tmp[2])
            if len(tmp) > 0 and tmp[0] == 'print_dump':
                dump = float(tmp[2])
                break
        return dump * dt

    # 读入当前时间步的拉格朗日点集合和受力
    def read_LagForce(self, timestep):
        # t=0初始化
        if timestep == 0:
            vertex=open('swimmer.vertex', 'r')
            n = int(vertex.readline())
            Lag = []
            for i in range(n):
                tmp=vertex.readline().split()
                Lag.append([float(tmp[0]), float(tmp[1])])
            LagXForce=[0]*n
            LagYForce=[0]*n
            vertex.close()
            return Lag, LagXForce, LagYForce

        else:
            yForcehead = 'fY_Lag.'
            xForcehead = 'fX_Lag.'
            tail = '.vtk'
            tag = str(timestep)
            while len(tag) < 4:
                tag = '0' + tag
            Xpointer = xForcehead + tag + tail
            Ypointer = yForcehead + tag + tail
            fx = open('hier_IB2d_data\\'+Xpointer, "r")
            fy = open('hier_IB2d_data\\'+Ypointer, "r")
            for i in range(5):
                fx.readline()
                fy.readline()
            fy.readline()
            tmp = fx.readline()
            Ntotal = int(tmp.split()[1])

            Lag = []  # 所有拉格朗日点的坐标
            for i in range(Ntotal):
                fy.readline()
                tmp = fx.readline()
                tmp2 = tmp.split()
                Lag.append([float(tmp2[0]), float(tmp2[1])])
            for i in range(5):
                fx.readline()
                fy.readline()
            LagXForce = []  # 每个拉格朗日点上的力
            LagYForce = []
            for i in range(Ntotal):
                LagXForce.append(float(fx.readline()))
                LagYForce.append(float(fy.readline()))
            fx.close()
            fy.close()

            return Lag, LagXForce, LagYForce

    # 计算当前物体质心
    def calculate_mass_center(self, Lag):
        n = len(Lag)
        x = 0.0
        y = 0.0
        for i in range(n):
            x += Lag[i][0]
            y += Lag[i][1]
        x = x/float(n)
        y = y/float(n)
        return [x, y]

    # 计算对心力矩
    def calculate_torque(self, Lag, center, fx, fy):
        n = len(Lag)
        torque = 0
        for i in range(n):
            torque += ((Lag[i][0]-center[0])*fy[i]-(Lag[i][1]-center[1])*fx[i])
        return torque

    # 计算总受力
    def calculate_force(self, Lag, fx, fy):
        n = len(Lag)
        FX = 0
        FY = 0
        for i in range(n-1):
            ds_squre = (Lag[i][0]-Lag[i+1][0])**2+(Lag[i][1]-Lag[i+1][1])**2
            ds = math.sqrt(ds_squre)
            FX += (fx[i]+fx[i+1])/2*ds
            FY += (fy[i]+fy[i+1])/2*ds
        return [FX, FY]

    # 输入action后根据当前的流场信息求解下一步的流场并更新当前的流场状态
    # 并返回s_:下一时刻的物体受力、质心坐标、对心力矩,reward:正相关于推进的距离
    # 负相关于质心y方向偏离可以接受的范围的距离
    def step(self, action):
        # 修改update文件中的kStiff
        RLfile = open('update_nonInv_Beams.m', 'r')
        RLtarget = open('update_nonInv_Beams.m.bak', 'w+')
        lines = RLfile.readlines()
        RLfile.close()
        for line in lines:
            tmp = line.split()
            if len(tmp) == 0 or tmp[0] != 'kStiff':
                RLtarget.write(line)
            else:
                k = float(tmp[2])
                if self.currentTime == 0:
                    self.k_beam = k
                if action == 'plus5': # 将update_nonInv_beam中的刚度增加5%
                    k *= 1.05
                if action == 'plus10': # 将update_nonInv_beam中的刚度增加10%
                    k *= 1.1
                if action == 'minus5':  # 将update_nonInv_beam中的刚度减小5%
                    k *= 0.95
                if action == 'minus10': # 将update_nonInv_beam中的刚度减小10%
                    k *= 0.9
                RLtarget.write('kStiff = ' + str(k) + ' ;\n')
        RLtarget.close()
        os.remove(RLfile.name)
        newname = RLtarget.name.replace('.bak', '')
        os.rename(RLtarget.name, newname)

        # 修改重启设置中的最新步
        RLfile = open('help_Me_Restart.m', 'r')
        RLtarget = open('help_Me_Restart.m.bak', 'w+')
        lines = RLfile.readlines()
        RLfile.close()
        for line in lines:
            tmp = line.split()
            if len(tmp) == 0 or tmp[0] != 'ctsave':
                RLtarget.write(line)
            else:
                RLtarget.write('ctsave = ' + str(self.currenTimeStep) + ' ;\n')
        RLtarget.close()
        os.remove(RLfile.name)
        newname = RLtarget.name.replace('.bak', '')
        os.rename(RLtarget.name, newname)

        # 修改input2d中的重启参数和终止参数
        RLfile = open('input2d', 'r')
        RLtarget = open('input2d.bak', 'w+')
        lines = RLfile.readlines()
        RLfile.close()
        for line in lines:
            tmp = line.split()
            if len(tmp) >0 and tmp[0] == 'Restart_Flag' and self.currentTime>0:
                RLtarget.write('Restart_Flag = 1\n')
            elif len(tmp) >0 and tmp[0] == 'Tfinal':
                #print('currenTime='+str(self.currentTime)+'正在将tfinal修改为'+str(self.currentTime+self.dt)+'\n')
                RLtarget.write('Tfinal = '+str(self.currentTime+self.dt+0.001)+'\n')
            else:
                RLtarget.write(line)
        RLtarget.close()
        os.remove(RLfile.name)
        newname = RLtarget.name.replace('.bak', '')
        os.rename(RLtarget.name, newname)



        """RLfile = open('input2d', 'r')
        lines = RLfile.readlines()
        RLfile.close()
        for line in lines:
            tmp = line.split()
            if len(tmp) == 0 or tmp[0] != 'Tfinal':
                continue
            else:
                tfinal=float(tmp[2])
                break"""
        #print('currenTime='+str(self.currentTime)+' ,tfinal='+str(tfinal)+'\n')


        # 调用matlab求解
        #eng = matlab.engine.start_matlab()
        self.eng.main2d(nargout=0)
        #eng.quit()

        # 时间步加一
        self.currenTimeStep += 1
        self.currentTime += self.dt

        # 判断终止
        if self.currentTime >= self.endTime:
            done = True
            self.eng.quit() ################
            # 到了终止后还需要改名结果文件夹的名字，标记是第几次训练
            os.rename('viz_IB2d', 'viz_IB2d_' + str(self.episode))
            os.rename('hier_IB2d_data', 'hier_IB2d_data_' + str(self.episode))

            # 还要把input2d中的重启标志改回0，终止时间改回初始值
            RLfile = open('input2d', 'r')
            RLtarget = open('input2d.bak', 'w+')
            lines = RLfile.readlines()
            RLfile.close()
            for line in lines:
                tmp = line.split()
                if len(tmp) > 0 and tmp[0] == 'Restart_Flag':
                    RLtarget.write('Restart_Flag = 0\n')
                elif len(tmp) > 0 and tmp[0] == 'Tfinal':
                    RLtarget.write('Tfinal = ' + str(self.endTime) + '\n')
                else:
                    RLtarget.write(line)
            RLtarget.close()
            os.remove(RLfile.name)
            newname = RLtarget.name.replace('.bak', '')
            os.rename(RLtarget.name, newname)

            #还要把update中的初值改回去
            RLfile = open('update_nonInv_Beams.m', 'r')
            RLtarget = open('update_nonInv_Beams.m.bak', 'w+')
            lines = RLfile.readlines()
            RLfile.close()
            for line in lines:
                tmp = line.split()
                tmp = line.split()
                if len(tmp) == 0 or tmp[0] != 'kStiff':
                    RLtarget.write(line)
                else:
                    RLtarget.write('kStiff = ' + str(self.k_beam) + ' ;\n')
            RLtarget.close()
            os.remove(RLfile.name)
            newname = RLtarget.name.replace('.bak', '')
            os.rename(RLtarget.name, newname)

        else:
            done = False

        # 设置奖励函数和更新环境状态
        preX=self.massCenter[0]
        preFX=self.Force[0]

        # 更新环境状态
        if self.currentTime < self.endTime:
            self.Lagpoints, self.fx, self.fy = self.read_LagForce(self.currenTimeStep)  # 读入当前时间步的拉格朗日点和力
            self.massCenter = self.calculate_mass_center(self.Lagpoints)  # 计算质心坐标
            self.torque = self.calculate_torque(self.Lagpoints, self.massCenter, self.fx, self.fy)  # 计算对质心的力矩
            self.Force = self.calculate_force(self.Lagpoints, self.fx, self.fy)  # 计算总受力

        # 计算奖励（暂定奖励函数正比于推进距离和阻力减少值
        reward = 20*(self.massCenter[0]-preX) #+0.00001*(preFX-self.Force[0])

        # 返回observation
        s_ = [self.massCenter[0], self.massCenter[1], self.Force[0], self.Force[1], self.torque]

        return s_, reward, done




