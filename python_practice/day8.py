# 1. 定义基础类（动物）
class Animal:
    def __init__(self, name):
        self.name = name  # 实例属性
    
    def run(self):  # 实例方法
        print(f"{self.name} is running")

# 2. 定义子类（猫，继承Animal）
class Cat(Animal):
    def __init__(self, name, age):
        super().__init__(name)  # 调用父类构造方法
        self.age = age  # 子类新增属性
    
    def meow(self):  # 子类新增方法
        print(f"{self.name} (age {self.age}) is meowing")

# 3. 测试代码
if __name__ == "__main__":
    dog = Animal("Dog")
    dog.run()  # 输出：Dog is running
    
    cat = Cat("Mimi", 2)
    cat.run()  # 继承父类方法，输出：Mimi is running
    cat.meow()  # 子类方法，输出：Mimi (age 2) is meowing
