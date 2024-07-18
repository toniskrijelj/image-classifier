import image_classifier as ic
# import digit_images_loader
import sketch_images_loader
import labels

trainset, testset, labels = sketch_images_loader.load(labels.sketch_labels2)
save_file = 'sketch_model2'

if __name__ == "__main__":

    clf = ic.ImageClassifier('cuda', 1, 28, 28, labels)
    epochs = 20
    for epoch in range(epochs):
        loss, correct, wrong, correct_test, wrong_test = clf.train_loop(trainset, testset)
        print(f"Epoch:{epoch} loss is {loss}")
        print(f"Correct Training:{correct}, wrong:{wrong}")
        print(f"Correct Testing:{correct_test}, wrong:{wrong_test}")
        print()

    clf.save(save_file)
