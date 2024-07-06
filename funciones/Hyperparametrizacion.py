import numpy as np
from funciones import Guardar
from funciones import CTGANSynthesizerMod
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
def Hiper(Principal, Meta, RangoEpocas, RangoBatchSize, RangoDiscriminatorDim, RangoDiscriminatorDecay,
          RangoDiscriminatorLR, RangoDiscriminatorSteps, RangoEmbeddingDim, RangoGeneratorDim, RangoGeneratorDecay, RangoGeneratorLR,
          RangoPac, EarlyStoppingEspera, EarlyStoppingDisminucion, Limite, Lista = []): 
  MejoresParametros = Lista
  Combinaciones = []
  for Epocas in RangoEpocas:
    for BatchSize in RangoBatchSize:
      for DiscriminatorDim in RangoDiscriminatorDim:
        for DiscriminatorDecay in RangoDiscriminatorDecay:
          for DiscriminatorLR in RangoDiscriminatorLR:
            for DiscriminatorSteps in RangoDiscriminatorSteps:
              for EmbeddingDim in RangoEmbeddingDim:
                for GeneratorDim in RangoGeneratorDim:
                  for GeneratorDecay in RangoGeneratorDecay:
                    for GeneratorLR in RangoGeneratorLR:
                      for Pac in RangoPac:
                        Combinaciones.append((Epocas, BatchSize, DiscriminatorDim, DiscriminatorDecay,
                                            DiscriminatorLR, DiscriminatorSteps, EmbeddingDim, GeneratorDim,
                                          GeneratorDecay, GeneratorLR, Pac)) #Añadimos todas las combinaciones posibles.
  print("Combinaciones creadas / ", end = "")
  np.random.shuffle(Combinaciones) #Mezclamos las combinaciones para coger al azar.
  #La iteración es demasiado larga así que iremos guardando los datos poco a poco parando la ejecución manualmente
  #y así creando una especie de checkpoints por si algo falla en algún momento. Luego leemos el archivo pickle con los datos y
  #eliminamos de la lista Combinaciones los parámetros con los que ya entrenamos al modelo.
  MP = set([tuple(x.values())[:-3] for x in Lista]) #Cogemos solo los valores de los parámetros sin la puntuación.
  try:
    for p in MP & set(Combinaciones): #Eliminamos de la lista Combinaciones las combinaciones ya creadas.
      Combinaciones.remove(p)
    for i in Combinaciones:
      if len(MejoresParametros) == Limite: #Verificamos que no supere el límite de iteraciones establecido.
        break
      Synthesizer = CTGANSynthesizerMod( #Creamos el sintetizador con unos parámetros aleatorios.
          metadata=Meta,
          verbose = True,
          epochs = i[0],
          batch_size = i[1],
          discriminator_dim = i[2],
          discriminator_decay = i[3],
          discriminator_lr = i[4],
          discriminator_steps = i[5],
          embedding_dim = i[6],
          generator_dim = i[7],
          generator_decay = i[8],
          generator_lr = i[9],
          pac = i[10],
          EarlyStoppingEspera = EarlyStoppingEspera,
          EarlyStoppingDisminucion = EarlyStoppingDisminucion)
      print("Entrenando modelo / ", end = "")
      Synthesizer.fit(Principal)
      Sintetico = Synthesizer.sample(num_rows=200)
      Diagnostico = run_diagnostic(
          Principal,
          Sintetico,
          Meta,
          False)
      if Diagnostico.get_score() !=1: #Si los datos tienen algún fallo queremos que siga con la siguiente iteración.
        continue
      Calidad = evaluate_quality(
          Principal,
          Sintetico,
          Meta,
          False)
      print("Nº", len(MejoresParametros)+1, end = ", ")
      print("Puntuación:", Calidad.get_score())
      MejoresParametros.append({"Epocas": i[0],
                                "BatchSize": i[1],
                                "DiscriminatorDim": i[2],
                                "DiscriminatorDecay": i[3],
                                "DiscriminatorLR": i[4],
                                "DiscriminatorSteps": i[5],
                                "EmbeddingDim": i[6],
                                "GeneratorDim": i[7],
                                "GeneratorDecay": i[8],
                                "GeneratorLR": i[9],
                                "Pac": i[10],
                                "Puntuación": Calidad.get_score(),
                                "Losses": Synthesizer.get_loss_values(),
                                "LossesPlot": Synthesizer.get_loss_values_plot()})
    return MejoresParametros
  except BaseException as e:
    print("Ha fallado en el intento nº", len(MejoresParametros)+1)
    Guardar(MejoresParametros, "MejoresParametros") #Si algo falla en la iteración los queremos guardar igualmente.
    raise e