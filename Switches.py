class Switch:
    def __init__(self):
        pass

    def switch(self):
        raise NotImplementedError("Not implemented!")


class TwoFiveSwitch(Switch):
    """Implementation of Switch which follows a simple 2-5 ratio rule: Train G for 2 epochs, and D for 5."""

    def __init__(self):
        Switch.__init__(self)
        self.state = 0

    def switch(self):
        if self.state < 2:
            self.state += 1
            return "G"
        else:
            self.state += 1
            if self.state >= 7:
                self.state = 0
            return "D"


class AlwaysDSwitch(Switch):
    """Implementation of Switch which always trains the discriminator (for use with dynamic training)"""

    def __init__(self):
        Switch.__init__(self)

    def switch(self):
        return "D"


class AlwaysGSwitch(Switch):
    """Implementation of Switch which always trains the generator"""

    def __init__(self):
        Switch.__init__(self)

    def switch(self):
        return "G"
