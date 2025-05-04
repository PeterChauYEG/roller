from src.env.data.game import N_DICES, N_DICE_FACES
from src.env.dice_face import DiceFace
from src.env.game_enums import DiceType


class Dice:
    def __init__(self, i):
        self.n_dices = N_DICES
        self.n_faces = N_DICE_FACES

        self.faces = []
        self.type = DiceType.ATTACK

        self.generate_faces()
        self.generate_type(i)

    def generate_faces(self):
        for i in range(self.n_faces):
            face = DiceFace()
            self.faces.append(face)

    def generate_type(self, i):
        if i < self.n_dices // 2:
            self.type = DiceType.ATTACK
        else:
            self.type = DiceType.DEFENSE

    # getters
    def get_faces(self):
        return self.faces

    def get_face(self, i):
        return self.faces[i]

    def get_type(self):
        return self.type
