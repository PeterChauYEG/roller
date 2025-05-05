from src.env.data.game import N_DICES, N_DICE_FACES
from src.env.dice_face import DiceFace
from src.env.game_enums import DiceType
from src.env.trait_manager import TraitManager


class Dice:
    def __init__(self, index, trait_manager: TraitManager):
        self.trait_manager = trait_manager
        self.index = index

        self.n_dices = N_DICES
        self.n_faces = N_DICE_FACES
        self.type = DiceType.ATTACK

        self.faces = self.generate_faces()
        self.type = self.generate_type()

    def generate_faces(self):
        faces = []

        for face_i in range(self.n_faces):
            face = DiceFace(
                self.trait_manager.get_trait(self.index, face_i)
            )
            faces.append(face)

        return faces

    def generate_type(self):
        if self.index < self.n_dices // 2:
            return DiceType.ATTACK
        else:
            return DiceType.DEFENSE

    # getters
    def get_faces(self):
        return self.faces

    def get_face(self, i):
        return self.faces[i]

    def get_type(self):
        return self.type
