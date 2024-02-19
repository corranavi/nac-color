#Assumptions:
# 1.  4 Branch
# 2.  Feature branch assente
# 3.  MRI1 + MRI2


def define_model():
    input_list = []
    output_list = []
    res_list = []

    res_net = tf.keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3), weights="imagenet",
                                             pooling="avg")

    """for layer in res_net.layers:
        if 'conv2' in layer.name or 'conv3' in layer.name or 'conv4' in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True"""

    # Regularization for ResNet50
    regularizer = tf.keras.regularizers.l2(l2_reg)
    if l2_reg != 0:
        if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
            print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
            exit()

        for layer in res_net.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)

        model_json = res_net.to_json()
        tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
        res_net.save_weights(tmp_weights_path)
        res_net = tf.keras.models.model_from_json(model_json)
        res_net.load_weights(tmp_weights_path, by_name=True)
    # print(res_net.losses)

    #if "DWI" in branches_list: -----------------------------------------------------------------
    DWI1_input = tf.keras.Input(shape=(224, 224, 3), name="DWI_1")
    DWI2_input = tf.keras.Input(shape=(224, 224, 3), name="DWI_2")
    input_list.extend((DWI1_input, DWI2_input))

    DWI_res = tf.keras.Model(inputs=res_net.input, outputs=res_net.output, name='resnet50_DWI')
    DWI1_res = DWI_res(DWI1_input)
    DWI2_res = DWI_res(DWI2_input)
    res_list.extend((DWI1_res, DWI2_res))

    conc_features = tf.keras.layers.concatenate([DWI1_res, DWI2_res], name='conc_DWI')
    DWI_pred = tf.keras.layers.Dense(2, activation='softmax', name="DWI")(conc_features)
    output_list.append(DWI_pred)
    #if MULTI == 1:
    DWI_pred_multi_1 = tf.keras.layers.Dense(1, activation='linear', name="DWI_multi_1")(conc_features)
    DWI_pred_multi_2 = tf.keras.layers.Dense(1, activation='linear', name="DWI_multi_2")(conc_features)
    output_list.append(DWI_pred_multi_1)
    output_list.append(DWI_pred_multi_2)
    print("DWI branch added")

    #if "T2" in branches_list: ------------------------------------------------------------------
    T21_input = tf.keras.Input(shape=(224, 224, 3), name="T2_1")
    T22_input = tf.keras.Input(shape=(224, 224, 3), name="T2_2")
    input_list.extend((T21_input, T22_input))

    T2_res = tf.keras.Model(inputs=res_net.input, outputs=res_net.output, name='resnet50_T2')
    T21_res = T2_res(T21_input)
    T22_res = T2_res(T22_input)
    res_list.extend((T21_res, T22_res))

    conc_features = tf.keras.layers.concatenate([T21_res, T22_res], name='conc_T2')
    T2_pred = tf.keras.layers.Dense(2, activation='softmax', name="T2")(conc_features)
    output_list.append(T2_pred)
    #if MULTI == 1:
    T2_pred_multi_1 = tf.keras.layers.Dense(1, activation='linear', name="T2_multi_1")(conc_features)
    T2_pred_multi_2 = tf.keras.layers.Dense(1, activation='linear', name="T2_multi_2")(conc_features)
    output_list.append(T2_pred_multi_1)
    output_list.append(T2_pred_multi_2)
    print("T2 branch added")

    #if "DCE_peak" in branches_list: ---------------------------------------------------------
    DCEpeak1_input = tf.keras.Input(shape=(224, 224, 3), name="DCE_peak_1")
    DCEpeak2_input = tf.keras.Input(shape=(224, 224, 3), name="DCE_peak_2")
    input_list.extend((DCEpeak1_input, DCEpeak2_input))

    DCEpeak_res = tf.keras.Model(inputs=res_net.input, outputs=res_net.output, name='resnet50_DCE_peak')
    DCEpeak1_res = DCEpeak_res(DCEpeak1_input)
    DCEpeak2_res = DCEpeak_res(DCEpeak2_input)
    res_list.extend((DCEpeak1_res, DCEpeak2_res))

    conc_features = tf.keras.layers.concatenate([DCEpeak1_res, DCEpeak2_res], name='conc_DCE_peak')
    DCE_peak_pred = tf.keras.layers.Dense(2, activation='softmax', name="DCE_peak")(conc_features)
    output_list.append(DCE_peak_pred)
    #if MULTI == 1:
    DCE_peak_pred_multi_1 = tf.keras.layers.Dense(1, activation='linear', name="DCE_peak_multi_1")(
        conc_features)
    DCE_peak_pred_multi_2 = tf.keras.layers.Dense(1, activation='linear', name="DCE_peak_multi_2")(
        conc_features)
    output_list.append(DCE_peak_pred_multi_1)
    output_list.append(DCE_peak_pred_multi_2)
    print("DCE_peak branch added")

    #if "DCE_3TP" in branches_list:
    DCE3TP1_input = tf.keras.Input(shape=(224, 224, 3), name="DCE_3TP_1")
    DCE3TP2_input = tf.keras.Input(shape=(224, 224, 3), name="DCE_3TP_2")
    input_list.extend((DCE3TP1_input, DCE3TP2_input))

    DCE3TP_res = tf.keras.Model(inputs=res_net.input, outputs=res_net.output, name='resnet50_DCE_3TP')
    DCE3TP1_res = DCE3TP_res(DCE3TP1_input)
    DCE3TP2_res = DCE3TP_res(DCE3TP2_input)
    res_list.extend((DCE3TP1_res, DCE3TP2_res))

    conc_features = tf.keras.layers.concatenate([DCE3TP1_res, DCE3TP2_res], name='conc_DCE_3TP')
    DCE_3TP_pred = tf.keras.layers.Dense(2, activation='softmax', name="DCE_3TP")(conc_features)
    output_list.append(DCE_3TP_pred)
    #if MULTI == 1:
    DCE_3TP_pred_multi_1 = tf.keras.layers.Dense(1, activation='linear', name="DCE_3TP_multi_1")(conc_features)
    DCE_3TP_pred_multi_2 = tf.keras.layers.Dense(1, activation='linear', name="DCE_3TP_multi_2")(conc_features)
    output_list.append(DCE_3TP_pred_multi_1)
    output_list.append(DCE_3TP_pred_multi_2)
    print("DCE_3TP branch added")

    # if FEATURES == 1:
    #     Features_input = tf.keras.Input(shape=14, name="Features_input")
    #     input_list.append(Features_input)

    #     dense_1 = tf.keras.layers.Dense(28, activation="relu")
    #     x = dense_1(Features_input)
    #     # dense_2 = tf.keras.layers.Dense(4, activation="relu")(x)
    #     Features_pred = tf.keras.layers.Dense(2, activation='softmax', name="Features_prediction")(x)
    #     res_list.append(x)
    #     output_list.append(Features_pred)
    #     print("Features branch added")

    conc_features = tf.keras.layers.concatenate(res_list, name='conc_all')
    drop = tf.keras.layers.Dropout(dropout)(conc_features)
    dense = tf.keras.layers.Dense(FC, activation='relu')(drop)

    pCR_pred = tf.keras.layers.Dense(2, activation='softmax', name="pCR")(dense)
    output_list.append(pCR_pred)

    print(input_list)
    print(output_list)

    model = tf.keras.Model(
        inputs=[input_list],
        outputs=[output_list]
    )
    return model