import ast
import gzip
import pickle

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from joblib import dump, load
from data.make_dataset import make_dataset
from features.make_features import make_features
from model.main import make_model
from sklearn.metrics import accuracy_score

@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(task, input_filename, model_dump_filename):
    if task == "find_comic_name":
        pass
    elif task == "is_comic_video" :
        df = make_dataset(input_filename)
        X, y = make_features(df, task, train_mode=True, vectorizer_path="vectorizer/vectorizer.pkl")

        model = make_model(task)
        model.fit(X, y)

        with open(model_dump_filename, 'wb') as f:
            pickle.dump(model, f)
    else:
        df = make_dataset(input_filename)

        features_list, labels_list, _ = make_features(df, task)
        X, y = features_list, labels_list


        model = make_model(task)
        model.fit(X, y)

        with open(model_dump_filename + task + ".json", 'wb') as f:
            dump(model, f)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def test(task, input_filename, model_dump_filename, output_filename):
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/test.csv", help="File training data")
@click.option("--model_dump_filename", default="models/text_classification.pkl", help="File to load model from")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")

def predict(task, input_filename, model_dump_filename, output_filename):
    df = make_dataset(input_filename)
    if task == "find_comic_name" :
        #is comic
        X_is_comic = make_features(df, "is_comic_video", train_mode=False, vectorizer_path="vectorizer/vectorizer.pkl")[0]
        model_dump_filename = "models/is_comic_video.json"
        with open(model_dump_filename, 'rb') as f:
            model = pickle.load(f)
        predictions = model.predict(X_is_comic)
        df["is_comic_video_prediction"] = predictions
        #is name
        features_list, _, token_list = make_features(df, task)
        model_dump_filenamea = "models/is_name.json"

        with open(model_dump_filenamea, 'rb') as f:
            model = load(f)
        preds = model.predict(features_list)
        df["is_name_prediction"] = ""

        cpt = 0
        for index, row in df.iterrows():
            # Assign binary predictions to the new column
            df.at[index,"is_name_prediction"] = preds[cpt:cpt + len(ast.literal_eval(row["tokens"]))]
            cpt += len(ast.literal_eval(row["tokens"]))

        # Combining results
        y_pred = []
        for index, row in df.iterrows():
            if row["is_comic_video_prediction"] == 1:
                name = ""
                tokens = np.array(ast.literal_eval(row["tokens"]))
                for i, pred in enumerate(row["is_name_prediction"]):
                    # if pred == 1:
                    #     print(row["is_name_prediction"])
                    #     print(i)
                    #     print(pred)
                    #     print(tokens[i])

                    if pred == 1:
                        name += " " + tokens[i]
                y_pred.append(name)
            else:
                y_pred.append("None")
        df[task + "_prediction"] = y_pred
        df.to_csv(output_filename, index=False)


    if task == "is_comic_video":

        X = make_features(df, task, train_mode=False, vectorizer_path="vectorizer/vectorizer.pkl")[0]  # Only extract X

        with open(model_dump_filename, 'rb') as f:
            model = pickle.load(f)

        predictions = model.predict(X)

        df['predictions'] = predictions
        df.to_csv(output_filename, index=False)


    elif task == "is_name":
        features_list, _, token_list = make_features(df, task)
        #print(token_list)
        # Load the trained model
        model = load(model_dump_filename)
        preds = model.predict(features_list)

        df[task + "_prediction"] = ""

        cpt = 0
        for index, row in df.iterrows():
            # Assign binary predictions to the new column
            df.at[index, task + "_prediction"] = preds[cpt:cpt + len(ast.literal_eval(row["tokens"]))]
            cpt += len(ast.literal_eval(row["tokens"]))


        df.to_csv(output_filename, index=False)
        return df






@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(task, input_filename):
    df = make_dataset(input_filename)
    if task == "is_comin_name":
        df = make_dataset(input_filename)

        # Make features (tokenization, lowercase, stopwords, stemming...)
        X, y = make_features(df, task, train_mode=True, vectorizer_path="vectorizer/vectorizer.pkl")

        model = make_model()

        # Run k-fold cross validation. Print results
        return evaluate_model(model, X, y)

    elif task == "is_name" :
        # Make features (tokenization, lowercase, stopwords, stemming...)
        X, y, _ = make_features(df, task)

        # Object with .fit, .predict methods
        model = make_model(task)

        # Run k-fold cross validation. Print results
        return evaluate_model(model, X, y)

    else :
        # is comic
        X_is_comic = make_features(df, "is_comic_video", train_mode=False, vectorizer_path="vectorizer/vectorizer.pkl")[
            0]
        model_dump_filename = "models/is_comic_video.json"
        with open(model_dump_filename, 'rb') as f:
            model = pickle.load(f)
        predictions = model.predict(X_is_comic)
        df["is_comic_video_prediction"] = predictions
        # is name
        features_list, _, token_list = make_features(df, task)
        model_dump_filenamea = "models/is_name.json"

        with open(model_dump_filenamea, 'rb') as f:
            model = load(f)
        preds = model.predict(features_list)
        df["is_name_prediction"] = ""

        cpt = 0
        for index, row in df.iterrows():
            # Assign binary predictions to the new column
            df.at[index, "is_name_prediction"] = preds[cpt:cpt + len(ast.literal_eval(row["tokens"]))]
            cpt += len(ast.literal_eval(row["tokens"]))

        # Combining results
        y_pred = []
        y = []
        for index, row in df.iterrows():
            if row["is_comic_video_prediction"] == 1:
                name = ""
                tokens = np.array(ast.literal_eval(row["tokens"]))
                for i, pred in enumerate(row["is_name_prediction"]):

                    if pred == 1:
                        name += " " + tokens[i]
                y_pred.append(name)
            else:
                y_pred.append("None")
            if row["is_comic"] == 1:
                name = ""
                for i, pred in enumerate(row["is_name"]):
                    if pred == 1:
                        name += " " + row["tokens"][i]
                y.append(name)
            else:
                y.append("None")
        print(f"Global accuracy {100 * round(accuracy_score(y, y_pred), 2)}%")




def evaluate_model(model, X, y):
    # Scikit learn has function for cross validation
    scores = cross_val_score(model, X, y, scoring="accuracy")
    print(f"Got accuracy {100 * np.mean(scores)}%")

    return scores


cli.add_command(train)
cli.add_command(test)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
