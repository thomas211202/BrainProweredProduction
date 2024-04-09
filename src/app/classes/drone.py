from pyardrone import ARDrone, at
from time import sleep, time


class Drone(object):
    def __init__(self):
        self.d = ARDrone()
        self.d.navdata_ready.wait()

    def takeoff(self):
        """Start flying the drone."""
        self.d.takeoff()

    def land(self):
        """Land the drone."""
        self.d.land()

    def move(self, direction):
        """
        Move the drone forward or backward or rotate the drone for 't' seconds.
        The 'direction' argument can be:
            - 'forward': move forward
            - 'backward': move backward
            - 'rotate_right': rotate 90 degrees to the right
            - 'rotate_left': rotate 90 degrees to the left
            - 'takeoff': start flying
            - 'land': stop flying
        """
        if direction == 'forward':
            self.forward(t=2)
        elif direction == 'backward':
            self.backward(t=1)
        elif direction == 'rotate_right':
            self.cw(t=0.4)
        elif direction == 'rotate_left':
            self.ccw(t=0.4)
        elif direction == 'takeoff':
            self.takeoff()
        elif direction == 'land':
            self.land()
        elif direction == 'up':
            self.move_up()
        elif direction == 'down':
            self.move_down()
        else:
            raise ValueError(
                'Given direction {} not supported!'.format(direction))
        self.d.hover()

    def forward(self, t=0.3, s=0.1):
        t_end = time() + t
        while time() < t_end:
            self.d.move(forward=s)

    def backward(self, t=0.3, s=0.1):
        t_end = time() + t
        while time() < t_end:
            self.d.move(backward=s)

    def cw(self, t=0.08, s=0.8):
        t_end = time() + t
        while time() < t_end:
            self.d.move(cw=s)

    def ccw(self, t=0.08, s=0.8):
        t_end = time() + t
        while time() < t_end:
            self.d.move(ccw=s)

    def move_up(self, t=0.03, s=0.1):
        t_end = time() + t
        while time() < t_end:
            self.d.move(move_up=s)

    def move_down(self, t=0.03, s=0.1):
        t_end = time() + t
        while time() < t_end:
            self.d.move(move_down=s)

if __name__ == '__main__':
    try:
            DRONE = Drone()
            DRONE.takeoff()
            sleep(2)
            DRONE.move('forward')
            sleep(2)
            DRONE.land()
            sleep(2)
            quit()
    except Exception as e:
        print(e)
        DRONE = None
        # handle_signint(1, 1)
