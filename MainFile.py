
from Model import Generator
import numpy as np

x = np.load( 'data/x.npy' )
y = np.load( 'data/y.npy' )

generator = Generator()

generator.batch_size( 50 )

generator.load_model( 'models/poem_1000_model_2.h5')

generator.fit( x , y , number_of_epochs=500 , plotTrainingHistory=False )
generator.save_model( 'models/poem_1500_model_2.h5')

print( generator.predict( input( 'Enter seed text : ') , seed=100  ) )





