# Tensorflow_with_Python
1. Tensorflow_Basic
> ver_1.x
>> [step01_tf_basic](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap01_Tensorflow_Basic/lecture/step01_tf_basic.py)
> ~~~ python3
> # python : 직접 실행 환경 / tensorflow : 간접 실행 환경
> ~~~
>> [step02_tf_variable_init](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap01_Tensorflow_Basic/lecture/step02_tf_variable_init.py)
> ~~~ python3
> # 변수 정의와 초기화
> # 상수 : 수정 불가, 초기화 필요 없음
> # 변수 : 수정 가능, 최가화 필요 있음
> # graph = node(연산자:+-*/) + edge(데이터:x,y)
> # tensor : 데이터의 자료구조(scala(0), vector(1), matrix(2), array(3), n-array)
> # add = tf.add(y, 20, name = "add") :  y = 50, y = 20 : 별도의 인수를 
> # 지정하지 않으면 자동으로 x, y로 지정
> ~~~
>> [step03_tensorboar](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap01_Tensorflow_Basic/lecture/step03_tensorboard.py)  
>> [step04_tensorboard2](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap01_Tensorflow_Basic/lecture/step04_tensorboard2.py)  
>> [step05_variable_type](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap01_Tensorflow_Basic/lecture/step05_variable_type.py)  
>> [step06_variable_assign](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap01_Tensorflow_Basic/lecture/step06_variable_assign.py)  
>> [step07_variable_feed](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap01_Tensorflow_Basic/lecture/step07_variable_feed.py)  
>> [step08_variable_feed_csv](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap01_Tensorflow_Basic/lecture_1x/step08_variable_feed_csv.py)  
>> [step09_tf_logic](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap01_Tensorflow_Basic/lecture_1x/step09_tf_logic.py)  
> 
> ver_2.x
>> [step01_eager_execution](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap01_Tensorflow_Basic/lecture_2x/step01_eager_execution.py)  
>> [step02_function_replace](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap01_Tensorflow_Basic/lecture_2x/step02_function_replace.py)  
>> [step03_@tf.function](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap01_Tensorflow_Basic/lecture_2x/step03_%40tf.function.py)  
>> [step04_@tf.function2](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap01_Tensorflow_Basic/lecture_2x/step04_%40tf.function2.py)  
>> [step05_ver1x_ver2](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap01_Tensorflow_Basic/lecture_2x/step05_ver1x_ver2.py)  
>
2. Tensor_Handling
>> [step01_shape_rank_size](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap02_Tensor_Handling/lecture/step01_shape_rank_size.py)  
>> [step02_convert_tensors](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap02_Tensor_Handling/lecture/step02_convert_tensors.py)  
>> [step03_arithFunction](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap02_Tensor_Handling/lecture/step03_arithFunction.py)  
>> [step04_linalgFunction](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap02_Tensor_Handling/lecture/step04_linalgFunction.py)  
>> [step05_transform01_reshape](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap02_Tensor_Handling/lecture/step05_transform01_reshape.py)  
>> [step05_transform02_squeeze](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap02_Tensor_Handling/lecture/step05_transform02_squeeze.py)  
>> [step05_transform03_slice](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap02_Tensor_Handling/lecture/step05_transform03_slice.py)  
>> [step05_transform04_expand](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap02_Tensor_Handling/lecture/step05_transform04_expand.py)  
>> [step06_image_slice](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap02_Tensor_Handling/lecture/step06_image_slice.py)  
>> [step07_image_transpose](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap02_Tensor_Handling/lecture/step07_image_transpose.py)
>
3. Linear_Regression
> ver_1x
>> [step01_constant_reduce](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap03_Linear_Regression/lecture_1x/step01_constant_reduce.py)  
>> [step02_random_chart](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap03_Linear_Regression/lecture_1x/step02_random_chart.py)  
>> [step03_regression_formula](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap03_Linear_Regression/lecture_1x/step03_regression_formula.py)  
>> [step04_regression_formula2](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap03_Linear_Regression/lecture_1x/step04_regression_formula2.py)  
>> [step05_regression_model](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap03_Linear_Regression/lecture_1x/step05_regression_model.py)  
>> [step06_regression_model2](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap03_Linear_Regression/lecture_1x/step06_regression_model2.py)
>> [step07_hyper_parameter](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap03_Linear_Regression/lecture_1x/step07_hyper_parameter.py)
>> [step08_fullBatch_miniBatch](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap03_Linear_Regression/lecture_1x/step08_fullBatch_miniBatch.py)  
> ver_2x  
>> [step01_gradientTape](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap03_Linear_Regression/lecture_2x/step01_gradientTape.py)
> ~~~ python3
> # tf.cast(object, dtype) : object의 dtype 변경
> ~~~
>> [step02_gradientTape_model](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap03_Linear_Regression/lecture_2x/step02_gradientTape_moel.py)  
>> [step03_gradientTape_model2_iris](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap03_Linear_Regression/lecture_2x/step03_gradientTape_model2_iris.py)
>
4. Classification
> ver_1x
>> [step01_index_return](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap04_Classification/lecture_1x/step01_index_return.py)  
>> [step02_entropy](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap04_Classification/lecture_1x/step02_entropy.py)  
>> [step03_sigmoid_classifier](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap04_Classification/lecture_1x/step03_sigmoid_classifier.py)  
>> [step04_sigmoid_classifier_iris](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap04_Classification/lecture_1x/step03_sigmoid_classifier.py)  
>> [step05_softmax_classifier](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap04_Classification/lecture_1x/step05_softmax_classifier.py)  
>> [step06_softmax_classifier_iris](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap04_Classification/lecture_1x/step06_softmax_classifier_iris.py)  
>> [step07_softmax_MNIST](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap04_Classification/lecture_1x/step07_softmax_MNIST.py)  
> ver_2x
>> [step01_gradientTape_sigmoid](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap04_Classification/lecture_2x/step01_gradientTape_sigmoid.py)  
>> [step02_gradientTape_softmax](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap04_Classification/lecture_2x/step02_gradientTape_softmax.py)  
>
5. ANN_DNN
>> [step01_softmax_classifier_ANN](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap05_ANN_DNN/lecture/step01_softmax_classifier_ANN.py)  
>> [step02_softmax_classifier_DNN](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap05_ANN_DNN/lecture/step02_softmax_classifier_DNN.py)  
>> [step03_MNIST_DNN](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap05_ANN_DNN/lecture/step03_MNIST_DNN.py)  
>> [step04_TfidfSparseMatrix2](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap05_ANN_DNN/lecture/step04_TfidfSparseMatrix2.py)  
>> [step05_Tfidf_sigmoid_DNN](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap05_ANN_DNN/lecture/step05_Tfidf_sigmoid_DNN.py)  
>> ~~~python3
>> """
>>step03_MNIST_DNN.py
>>
>>DNN model + MNIST + Hyper parameters + Mini batch(온라인 배치)
>> 
>> - Network layer
>>   input nodes   : 28 * 28 = 784
>>   hidden1 nodes : 128 - 1층 ! 입력수가 많으면 노드수로 비례해서 많아짐
>>   hidden2 nodes : 64  - 2층
>>   ouput nodes   : 10  - 3층
>> 
>> - Hyper parameters
>>   lr    : 학습률
>>   epochs : 전체 dataset 재사용 횟수
>>   batch size : 1회 data 공급 횟수(mini batch)
>>   iter size  : 반복횟수
>>    -> 1epoch(60,00) : batch size(200) * iter size(300)
>>"""
>> ~~~
>
6. kearas_model 
> keras_DNN(저수준 API sklearn과 비슷함)
>> [step01_keras_iris_DNN](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap06_keras_model/lecture01_keras_DNN/step01_keras_iris_DNN.py)  
>> [step02_keras_mnist_DNN](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap06_keras_model/lecture01_keras_DNN/step02_keras_mnist_DNN.py)  
>> [step03_keras_mnist_DNN_history](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap06_keras_model/lecture01_keras_DNN/step03_keras_mnist_DNN_history.py)  
>> [step04_keras_mnist_DNN_history_dropout](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap06_keras_model/lecture01_keras_DNN/step04_keras_mnist_DNN_history_dropout.py)  
>> [step05_keras_mnist_DNN_EarlyStop](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap06_keras_model/lecture01_keras_DNN/step05_keras_mnist_DNN_EarlyStop.py)  
> kears_export(고수준 API)
>> [step01_tf_Dataset](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap06_keras_model/lecture02_keras_export/step01_tf_Dataset.py)  
>> [step02_gradientTape](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap06_keras_model/lecture02_keras_export/step02_gradientTape.py)  
>> [step03_export_DNN_model](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap06_keras_model/lecture02_keras_export/step03_export_DNN_model.py)  
>  
7. CNN
> ver_1x  
>> [step01_mnist_cnn_basic](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap07_CNN/lecture_1x/step01_mnist_cnn_basic.py)  
>> [step02_real_image_cnn_basic](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap07_CNN/lecture_1x/step02_real_image_cnn_basic.py)  
>> [step03_mnist_CNN_model](https://github.com/DominKim/Tensorflow_with_Python/blob/master/chap07_CNN/lecture_1x/step03_mnist_CNN_model.py)  # Tensorflow_DNN_CNN_RNN_Basic
>> [step04_mnist_CNN_name_scopre](https://github.com/DominKim/Tensorflow_DNN_CNN_RNN_Basic/blob/master/chap07_CNN/lecture_1x/step04_mnist_CNN_name_scopre.py)  
> ver_2x
>> [step01_keras_cifar10_CNN_model](https://github.com/DominKim/Tensorflow_DNN_CNN_RNN_Basic/blob/master/chap07_CNN/lecture_2x/step01_keras_cifar10_CNN_model.py)  
>> [step02_export_CNN_model](https://github.com/DominKim/Tensorflow_DNN_CNN_RNN_Basic/blob/master/chap07_CNN/lecture_2x/step02_export_CNN_model.py)  
> cat_dog_classifier
>> [lecture_3_cat_dog_classifier](https://github.com/DominKim/Tensorflow_DNN_CNN_RNN_Basic/tree/master/chap07_CNN/lecture_3_cat_dog_classifier)  
8. Celeb_CNN
>> [lecture01_image_crawling](https://github.com/DominKim/Tensorflow_DNN_CNN_RNN_Basic/tree/master/chap08_Celeb_CNN/lecture01_image_crawling)  
>> [lecture02_face_landmark](https://github.com/DominKim/Tensorflow_DNN_CNN_RNN_Basic/tree/master/chap08_Celeb_CNN/lecutre02_face_landmark)  
>> [lecture03_CNN_classifier](https://github.com/DominKim/Tensorflow_DNN_CNN_RNN_Basic/tree/master/chap08_Celeb_CNN/lecture03_CNN_classifier)  
9. word_embedding_RNN
>> [step01_text_vector](https://github.com/DominKim/Tensorflow_DNN_CNN_RNN_Basic/blob/master/chap09_word_embedding_RNN/lecture/step01_text_vector.py)  
>> [step02_features_extract](https://github.com/DominKim/Tensorflow_DNN_CNN_RNN_Basic/blob/master/chap09_word_embedding_RNN/lecture/step02_features_extract.py)  
>> [step03_sparse_matrix_classifier](https://github.com/DominKim/Tensorflow_DNN_CNN_RNN_Basic/blob/master/chap09_word_embedding_RNN/lecture/step03_sparse_matrix_classifier.py)  
>> [step04_word_embedding_LSTM](https://github.com/DominKim/Tensorflow_DNN_CNN_RNN_Basic/blob/master/chap09_word_embedding_RNN/lecture/step04_word_embedding_LSTM.py)  