import pickle
def Guardar(var, name):
  with open("../data/variables/" + name + ".pickle", "wb") as file:
    pickle.dump(var, file)
def Abrir(name):
  with open("../data/variables/" + name + ".pickle", "rb") as file:
    return pickle.load(file)