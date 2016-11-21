def main():
    # import  mnist_loader as mn
    # training_data, validation_data, test_data = mn.load_data_wrapper()
    # print(type(test_data))
    import loadMNIST as dl
    training_data, validation_data, test_data = dl.load_data()

    print "Datset Loaded"

    import nn
    net = nn.Network([784, 30, 10])
    print "network done"
    accuracy = net.SGD(training_data, 1, 25, 3.0,test_data=test_data)
    print "Accuracy " + str(accuracy)
    # import network
    # net = network.Network([784, 30, 10])
    # net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

main()