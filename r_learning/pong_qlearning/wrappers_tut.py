
class Env:
    def __init__(self):
        self.num = 2

    def step(self, num):
        print('step of Env')
        return num + 3


class A(Env):
    def __init__(self, env):
        self.env = env
        self.num = env.num

    def step(self, num):
        print('step of A')
        return self.env.step(self.env.num)


class O(A):
    def step(self, num):
        print('step of O')
        return self.obs(self.env.step(num))

    def obs(self):
        return


class B(O):
    def __init__(self, env):
        super().__init__(env)

    def obs(self, num):
        print('obs of B')
        return num * 4


class C(O):
    def __init__(self, env):
        super().__init__(env)

    def obs(self, obs):
        print('obs of C')
        return obs * 6


env = Env()
env = B(env)
env = C(env)
print(env.step(2))