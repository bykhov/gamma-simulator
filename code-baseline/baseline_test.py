from unet import unet_model4, unet_model3, unet_model2
from utils import load_train_data, load_test_data, save_test_data
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# %% Load data
data_directory = 'D:/Google Drive/Research/2 Activity/database/'
X_train, Y_train, sample_length = load_train_data(data_directory=data_directory)

# %% Train model
learning_rate = 0.001
print('frame_length:', sample_length)

tf.keras.backend.clear_session()
model = unet_model3(frame_length=sample_length)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])

callback_stp = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                patience=5,
                                                verbose=2,
                                                restore_best_weights=True)

callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.5, patience=1, min_lr=5e-5, verbose=1)

history = model.fit(X_train, Y_train, batch_size=256, epochs=120, validation_split=0.1,
                    callbacks=[callback_stp, callback_reduce_lr], verbose=1)

# %% Test model
perf = pd.DataFrame(columns=['k', 'Acc'])

cnt2 = 0
for k in range(10, 70, 10):
    X_test, Y_test = load_test_data(k)

    print('----', k, '-----')
    Y_pred = model.predict(X_test)
    test_accuracy = model.evaluate(X_test, Y_test)[-1]
    print('test acc:', test_accuracy)

    perf.loc[cnt2] = [k / 100, test_accuracy]
    cm = confusion_matrix(Y_test.reshape(-1), np.argmax(Y_pred, axis=-1).reshape(-1),
                          normalize=None, labels=[0, 1, 2, 3, 4, 5])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4, 5])
    disp.plot(cmap=plt.cm.Blues)
    disp.ax_.set_title('Confusion matrix')
    plt.savefig('confusion_matrix_' + str(k) + '.pdf')

    # sns_plot = sns.heatmap(cm, fmt='d', annot=True, square=True,
    #                        cmap='gray_r', vmin=0, vmax=0,  # set all to white
    #                        linewidths=0.5, linecolor='k',  # draw black grid lines
    #                        cbar=False)                     # disable colorbar
    # sns.despine(left=False, right=False, top=False, bottom=False)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.savefig('confusion_matrix_' + str(k) + '.pdf')

    cnt2 += 1

perf.to_csv('perf.csv')
# Save to plot
plt.rcParams['text.usetex'] = True
fig = perf.plot(x='k', y='Acc', xlabel=r'Activity $[\lambda]$', ylabel='Measured activity ratio', legend=False,
                grid=True).get_figure()
fig.savefig('performance.pdf')
