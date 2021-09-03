# Imports


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# Functions and methods


def add_date(dataframe):

    """
    Function adding date-time and date to the dataframe from timestamp
    :param dataframe: dataframe with timestamp column (timestamp in seconds)
    :return: dataframe with two new columns
    """

    new_df = dataframe
    new_df['date_clock'] = pd.to_datetime(new_df['timestamp'], unit='s')
    new_df['date'] = new_df['date_clock'].dt.date

    return new_df


def get_all_users(events):

    """
    Function getting all unique users from events dataframe
    :param events: dataframe with events
    :return: dataframe with unique users_id
    """

    users_id = events["user_id"].unique()
    users_id.sort()
    users_id = pd.DataFrame(users_id, columns=["user_id"])

    return users_id


def get_pivot_table(dataframe, columns, index="user_id", values="step_id", aggfunc="count", fill_value=0):

    """
    Custom function for pivot tables
    :param dataframe: dataframe
    :param columns: column of interest
    :param index: column to group by
    :param values: column to aggregate
    :param aggfunc: aggregate function
    :param fill_value: value to replace missing values with
    :return: pivot table dataframe
    """

    new_df = dataframe
    new_df = new_df.pivot_table(index=index,
                                columns=columns,
                                values=values,
                                aggfunc=aggfunc,
                                fill_value=fill_value).reset_index()

    return new_df


def get_scores(submissions, users_id):

    """
    Function getting scores for all unique users_id
    :param submissions: dataframe with submissions status
    :param users_id: unique users_id
    :return: dataframe with scores for all unique users_id
    """

    scores = get_pivot_table(submissions, "submission_status")
    scores_dataframe = users_id.merge(scores, on='user_id', how='outer')
    scores_dataframe = scores_dataframe.fillna(0)

    return scores_dataframe


def add_pass_mark(train_dataframe, threshold):

    """
    Function adding pass mark to dataframe with data about users' scores
    :param train_dataframe: dataframe with data about users' scores
    :param threshold: threshold for passing course
    :return: marked dataframe
    """

    new_df = train_dataframe
    new_df['passed_course'] = new_df.correct > threshold

    return new_df


def get_filtering_timestamp(events, threshold):

    """
    Function adding information about threshold for users
    :param events: dataframe with events
    :param threshold: threshold for time limit to predict
    :return: dataframe with filtering timestamp
    """

    users_start_time = events.groupby("user_id", as_index=False) \
        .agg({"timestamp": "min"}) \
        .rename({"timestamp": "first_timestamp"}, axis=1)

    users_start_time["user_learning_time_threshold"] = \
        users_start_time.user_id.map(str) + "_" + \
        (users_start_time.first_timestamp + threshold).map(str)

    users_start_time = users_start_time.drop(columns=["first_timestamp"], axis=1)

    return users_start_time


def get_time_features(events):

    """
    Function getting time features from events dataframe
    :param events: dataframe with events
    :return: dataframe with time features
    """

    time_features = events.groupby("user_id", as_index=False) \
        .agg({"timestamp": "min"}) \
        .rename({"timestamp": "start_timestamp"}, axis=1)

    time_features['date_clock'] = pd.to_datetime(time_features['start_timestamp'], unit='s')
    time_features['start_year'] = time_features['date_clock'].dt.year
    time_features['start_quarter'] = time_features['date_clock'].dt.quarter
    time_features['start_month'] = time_features['date_clock'].dt.month
    time_features['start_week'] = time_features['date_clock'].dt.isocalendar().week
    time_features['start_day'] = time_features['date_clock'].dt.day
    time_features['start_day_of_week'] = time_features['date_clock'].dt.weekday
    time_features['start_hour'] = time_features['date_clock'].dt.hour

    time_features = time_features.drop(columns=["start_timestamp", "date_clock"], axis=1)

    return time_features


def filter_by_time(dataframe, users_start_time):

    """
    Function filtering dataframe by time threshold
    :param dataframe: dataframe
    :param users_start_time: dataframe with filtering timestamp
    :return: filtered dataframe
    """

    new_df = dataframe
    new_df["user_time"] = new_df.user_id.map(str) + "_" + new_df.timestamp.map(str)
    new_df = new_df.merge(users_start_time, on="user_id", how="outer")
    new_df = new_df[new_df.user_time <= new_df.user_learning_time_threshold]

    return new_df


def get_steps_tried(train_submissions):

    """
    Function getting steps tried from filtered submission dataframe
    :param train_submissions: filtered submission dataframe
    :return: dataframe with steps tried
    """

    steps_tried = train_submissions.groupby("user_id", as_index=False). \
        step_id.nunique().rename(columns={"step_id": "steps_tried"})

    return steps_tried


def get_unique_days(dataframe):

    """
    Function getting unique days of users from dataframe
    :param dataframe: dataframe with date
    :return: dataframe with unique days
    """

    days = dataframe.groupby('user_id').date.nunique().to_frame().reset_index()

    return days


def get_x_y_train(events, submissions, threshold):

    """
    Function getting x and y for training models from events and submission dataframes
    :param events: dataframe with events
    :param submissions: dataframe with submission status
    :param threshold: time threshold for prediction
    :return: x and y for training models
    """

    # Копируем датафреймы
    new_events = events
    new_submissions = submissions

    # Lобавляем дату и время из временных меток
    new_events = add_date(new_events)
    new_submissions = add_date(new_submissions)

    # Получаем балы пользователей и помечаем тех кто прошел курс
    users_id = get_all_users(new_events)
    marked_dataframe = get_scores(new_submissions, users_id)
    marked_dataframe = add_pass_mark(marked_dataframe, 40)  # 40 балов - курс пройден

    # Получаем время начала курса каждым пользователем и фильтруем записи по порогу времени из условия
    users_start_time = get_filtering_timestamp(new_events, threshold)
    event_data_train = filter_by_time(new_events, users_start_time)
    submission_data_train = filter_by_time(new_submissions, users_start_time)

    # Получаем количество попыток пользователей решить задания
    steps_tried = get_steps_tried(submission_data_train)

    # Получаем количество различных действий пользователей
    actions = get_pivot_table(event_data_train, "action")
    status = get_pivot_table(submission_data_train, "submission_status")

    # Получаем количесвто уникальных дней пользователей и временные фичи
    time_features = get_time_features(new_events)
    user_days_events = get_unique_days(event_data_train)
    user_days_submissions = get_unique_days(submission_data_train)

    # Создаем x_train
    x = steps_tried
    x = x.merge(status, on="user_id", how="outer")
    x = x.merge(actions, on="user_id", how="outer")
    x = x.merge(time_features, on="user_id", how="outer")
    x = x.merge(marked_dataframe[["user_id", "passed_course"]], on="user_id", how="outer")
    x = x.merge(user_days_events, on="user_id", how="outer").rename({"date": "e_days"}, axis=1)
    x = x.merge(user_days_submissions, on="user_id", how="outer").rename({"date": "s_days"}, axis=1)

    # Создаем y_train
    y = x.passed_course
    y = y.map(int)

    # Убираем лишние данные из x_train и заполняем NaN
    x = x.fillna(0)
    x = x.drop(["passed_course"], axis=1)
    x = x.set_index(x.user_id).drop("user_id", axis=1)

    return x, y


def get_x_pred(events, submissions):

    """
    Function getting x for testing models from events and submission dataframes
    :param events: dataframe with events
    :param submissions: dataframe with submission status
    :return: x for testing models
    """

    # Копируем датафреймы
    new_events = events
    new_submissions = submissions

    # Добавляем дату и время из временных меток
    new_events = add_date(new_events)
    new_submissions = add_date(new_submissions)

    # Получаем количество попыток пользователей решить задания
    steps_tried = get_steps_tried(new_submissions)

    # Получаем количество различных действий пользователей
    actions = get_pivot_table(new_events, "action")
    status = get_pivot_table(new_submissions, "submission_status")

    # Получаем количесвто уникальных дней пользователей и временные фичи
    time_features = get_time_features(new_events)
    user_days_events = get_unique_days(new_events)
    user_days_submissions = get_unique_days(new_submissions)

    # Создаем x_pred
    x = steps_tried
    x = x.merge(status, on="user_id", how="outer")
    x = x.merge(actions, on="user_id", how="outer")
    x = x.merge(time_features, on="user_id", how="outer")
    x = x.merge(user_days_events, on="user_id", how="outer").rename({"date": "e_days"}, axis=1)
    x = x.merge(user_days_submissions, on="user_id", how="outer").rename({"date": "s_days"}, axis=1)

    # Убираем лишние данные из x_pred и заполняем NaN
    x = x.fillna(0)
    x = x.set_index(x.user_id).drop("user_id", axis=1)
    x = x.sort_index()

    return x


def random_forest_classifier(x_train, x_test, y_train, y_test, scaler, cv=5):

    """
    Function creating pipeline for random forest classifier, fit the pipeline and evaluate it with roc_auc scores
    :param x_train: x for training
    :param x_test: x for testing
    :param y_train: y for training
    :param y_test: y for testing
    :param scaler: scaler
    :param cv: number of splits for cross validation
    :return: pipeline, test and train scores
    """

    params = {
        "n_estimators": range(10, 1000, 10),
        "criterion": ["gini", "entropy"],
        "max_depth": range(5, 50, 5),
        "min_samples_leaf": range(5, 50, 5),
        "min_samples_split": range(5, 50, 5),
        "max_features": ["auto", "sqrt", "log2"],
        "class_weight": ["balanced", "balanced_subsample"],
        "bootstrap": [True, False]
    }

    rtc = RandomForestClassifier(n_jobs=-1, random_state=42)
    clf = RandomizedSearchCV(rtc, cv=cv, scoring="roc_auc", param_distributions=params, n_jobs=-1)
    pipeline = make_pipeline(scaler, clf)
    pipeline.fit(x_train, y_train)

    train_score = pipeline.score(x_train, y_train)
    test_score = pipeline.score(x_test, y_test)

    return pipeline, train_score, test_score


def gradient_boosting_classifier(x_train, x_test, y_train, y_test, scaler):

    """
    Function creating pipeline for gradient boosting classifier, fit the pipeline and evaluate it with roc_auc scores
    :param x_train: x for training
    :param x_test: x for testing
    :param y_train: y for training
    :param y_test: y for testing
    :param scaler: scaler
    :return: pipeline, test and train scores
    """

    gbc = GradientBoostingClassifier()
    pipeline = make_pipeline(scaler, gbc)
    pipeline.fit(x_train, y_train)

    train_score = roc_auc_score(y_train, pipeline.predict_proba(x_train)[:, 1])
    test_score = roc_auc_score(y_test, pipeline.predict_proba(x_test)[:, 1])

    return pipeline, train_score, test_score


def linear_regression_classifier(x_train, x_test, y_train, y_test, scaler, cv=5):

    """
    Function creating pipeline for linear regression classifier, fit the pipeline and evaluate it with roc_auc scores
    :param x_train: x for training
    :param x_test: x for testing
    :param y_train: y for training
    :param y_test: y for testing
    :param scaler: scaler
    :param cv: number of splits for cross validation
    :return: pipeline, test and train scores
    """

    params = {
        "penalty": ["l1", "l2", "elasticnet", "none"],
        "C": np.linspace(0.1, 10, 100),
        "fit_intercept": [True, False],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "max_iter": range(100, 1000, 10),
        "class_weight": ["balanced", None]
    }

    lrc = LogisticRegression(n_jobs=-1, random_state=42)
    clf = RandomizedSearchCV(lrc, cv=cv, scoring="roc_auc", param_distributions=params, n_jobs=-1)
    pipeline = make_pipeline(scaler, clf)
    pipeline.fit(x_train, y_train)

    train_score = pipeline.score(x_train, y_train)
    test_score = pipeline.score(x_test, y_test)

    return pipeline, train_score, test_score


def decision_tree_classifier(x_train, x_test, y_train, y_test, scaler, cv=5):

    """
    Function creating pipeline for decision tree classifier, fit the pipeline and evaluate it with roc_auc scores
    :param x_train: x for training
    :param x_test: x for testing
    :param y_train: y for training
    :param y_test: y for testing
    :param scaler: scaler
    :param cv: number of splits for cross validation
    :return: pipeline, test and train scores
    """

    params = {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": range(2, 50, 1),
        "min_samples_split": range(2, 50, 1),
        "min_samples_leaf": range(2, 50, 1),
        "max_features": ["auto", "sqrt", "log2"],
        "class_weight": ["balanced", None]
    }

    dtc = DecisionTreeClassifier(random_state=42)
    clf = RandomizedSearchCV(dtc, cv=cv, scoring="roc_auc", param_distributions=params, n_jobs=-1)
    pipeline = make_pipeline(scaler, clf)
    pipeline.fit(x_train, y_train)

    train_score = pipeline.score(x_train, y_train)
    test_score = pipeline.score(x_test, y_test)

    return pipeline, train_score, test_score


def naive_bayes_classifier(x_train, x_test, y_train, y_test, scaler):

    """
    Function creating pipeline for naive bayes classifier, fit the pipeline and evaluate it with roc_auc scores
    :param x_train: x for training
    :param x_test: x for testing
    :param y_train: y for training
    :param y_test: y for testing
    :param scaler: scaler
    :return: pipeline, test and train scores
    """

    nbc = GaussianNB()
    pipeline = make_pipeline(scaler, nbc)
    pipeline.fit(x_train, y_train)

    train_score = roc_auc_score(y_train, pipeline.predict_proba(x_train)[:, 1])
    test_score = roc_auc_score(y_test, pipeline.predict_proba(x_test)[:, 1])

    return pipeline, train_score, test_score


def neural_network_classifier(x_train, x_test, y_train, y_test, scaler):

    """
    Function creating pipeline for neural network classifier, fit the pipeline and evaluate it with roc_auc scores
    :param x_train: x for training
    :param x_test: x for testing
    :param y_train: y for training
    :param y_test: y for testing
    :param scaler: scaler
    :return: pipeline, test and train scores
    """

    mplc = MLPClassifier(hidden_layer_sizes=(16, 6), learning_rate="adaptive", activation="tanh", max_iter=1000)
    pipeline = make_pipeline(scaler, mplc)
    pipeline.fit(x_train, y_train)

    train_score = roc_auc_score(y_train, pipeline.predict_proba(x_train)[:, 1])
    test_score = roc_auc_score(y_test, pipeline.predict_proba(x_test)[:, 1])

    return pipeline, train_score, test_score
