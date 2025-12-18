from mycelium import activation

a = dict([
    (name, cls) for name, cls in activation.__dict__.items()
    if isinstance(cls, type)
])


from unittest import TestCase

class Car:

    def __init__(self, speed=0):
        self.speed = speed
        self.odometer = 0
        self.time = 0

    def accelerate(self):
        self.speed += 5

    def brake(self):
        self.speed -= 5

    def step(self):
        self.odometer += self.speed
        self.time += 1

    def average_speed(self):
        if self.time != 0:
            avg_speed = self.odometer / self.time
        else:
            avg_speed = 0
        return avg_speed


if __name__ == '__main__':

    my_car = Car()
    print("I'm a car!")
    while True:
        action = input("What should I do? [A]ccelerate, [B]rake, "
                        "show [O]dometer, or show average [S]peed?").upper()
        if action not in "ABOS" or len(action) != 1:
            print("I don't know how to do that")
            continue
        if action == 'A':
            my_car.accelerate()
            print("Accelerating...")
        elif action == 'B':
            my_car.brake()
            print("Braking...")
        elif action == 'O':
            print("The car has driven {} kilometers".format(my_car.odometer))
        elif action == 'S':
            print("The car's average speed was {} kph".format(my_car.average_speed()))
        my_car.step()



class TestCar(TestCase):
    def setUp(self):
        self.car = Car()


class TestInit(TestCar):
    def test_initial_speed(self):
        self.assertEqual(self.car.speed, 0)

    def test_initial_odometer(self):
        self.assertEqual(self.car.odometer, 0)

    def test_initial_time(self):
        self.assertEqual(self.car.time, 0)

    def test_initial_average_speed(self):
        self.assertEqual(self.car.average_speed(), 0)


class TestAccelerate(TestCar):
    def test_accelerate_from_zero(self):
        self.car.accelerate()
        self.assertEqual(self.car.speed, 5)

    def test_multiple_accelerates(self):
        for _ in range(3):
            self.car.accelerate()
        self.assertEqual(self.car.speed, 15)


class TestBrake(TestCar):
    def test_brake_once(self):
        self.car.accelerate()
        self.car.brake()
        self.assertEqual(self.car.speed, 0)

    def test_multiple_brakes(self):
        for _ in range(5):
            self.car.accelerate()
        for _ in range(3):
            self.car.brake()
        self.assertEqual(self.car.speed, 10)

    def test_should_not_allow_negative_speed(self):
        self.car.brake()
        self.assertEqual(self.car.speed, 0)

    def test_multiple_brakes_at_zero(self):
        for _ in range(3):
            self.car.brake()
        self.assertEqual(self.car.speed, 0)

