from src.model_trainer import CovidClassifier

def main():
    clf = CovidClassifier(dataset_path='dataset')
    clf.prepare_data()
    clf.build_model()
    clf.train(epochs=20)
    clf.evaluate()
    clf.save()

if __name__ == "__main__":
    main()
