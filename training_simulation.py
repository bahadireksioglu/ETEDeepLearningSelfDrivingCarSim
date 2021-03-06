print('Set-up')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from autonomous_car_simulation.utils import *
from sklearn.model_selection import train_test_split


# STEP 1
path = 'training_data'
data = import_data_info(path)

# STEP 2
data = balance_data(data, display=False)

# STEP 3
images_path, steerings = load_data(path, data)
#print(images_path[0], steerings[0])

# STEP 4
x_train, x_val, y_train, y_val = train_test_split(images_path, steerings, test_size=0.2, random_state=5)
print('Total Training Imgs : ', len(x_train))
print('Total Val Imgs : ', len(x_val))

# STEP 5

# STEP 6

# STEP 7

# STEP 8
model = create_model()
model.summary()

# STEP 9
history = model.fit(batch_generator(x_train, y_train, 100, 1), steps_per_epoch=300, epochs=10,
          validation_data=batch_generator(x_val, y_val, 100, 0), validation_steps=200)

# STEP 10
model.save('model.h5')
print('Model saved.')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0, 1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

