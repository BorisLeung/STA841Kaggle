from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

SEED = 662
MAX_ITER = 2000


def suggest_logistic_regression(
    trial: optuna.Trial, max_iter: int = MAX_ITER, seed: int = SEED
) -> LogisticRegression:
    C = trial.suggest_float("C", 1e-5, 1e5, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
    return LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,
        penalty="elasticnet",
        max_iter=max_iter,
        solver="saga",
        random_state=seed,
    )


def suggest_linear_svc(
    trial: optuna.Trial, max_iter: int = MAX_ITER, seed: int = SEED
) -> list[LinearSVC, CalibratedClassifierCV]:
    C = trial.suggest_float("C", 1e-5, 1e5, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    intercept_scaling = trial.suggest_float("intercept_scaling", 1e-10, 1e10, log=True)
    model = LinearSVC(
        C=C,
        penalty=penalty,
        intercept_scaling=intercept_scaling,
        max_iter=max_iter,
        random_state=SEED,
    )
    calibration_method = trial.suggest_categorical(
        "calibration_method", ["sigmoid", "isotonic"]
    )
    calibrated_model = CalibratedClassifierCV(
        model, cv="prefit", method=calibration_method
    )
    return [model, calibrated_model]


def suggest_kernel_svc(
    trial: optuna.Trial, max_iter: int = MAX_ITER, seed: int = SEED
) -> SVC:
    C = trial.suggest_float("C", 1e-5, 1e5, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    degree = trial.suggest_int("degree", 1, 5) if kernel == "poly" else 0
    gamma = (
        trial.suggest_categorical("gamma", ["scale", "auto"])
        if kernel != "linear"
        else "scale"
    )
    coef0 = trial.suggest_float("coef0", -1, 1) if kernel in ["poly", "sigmoid"] else 0
    return SVC(
        C=C,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        max_iter=max_iter,
        random_state=seed,
        probability=True,
    )


def suggest_naive_bayes(trial: optuna.Trial, seed: int = SEED) -> GaussianNB:
    return GaussianNB()


def suggest_knn_classifier(
    trial: optuna.Trial, seed: int = SEED
) -> KNeighborsClassifier:
    return KNeighborsClassifier(
        n_neighbors=trial.suggest_int("n_neighbors", 2, 10),
        weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
        algorithm=trial.suggest_categorical(
            "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
        ),
        leaf_size=trial.suggest_int("leaf_size", 10, 50),
        p=trial.suggest_int("p", 1, 2),
    )


def suggest_random_forest(
    trial: optuna.Trial, seed: int = SEED
) -> RandomForestClassifier:
    # model = RandomForestClassifier(
    #     max_depth=trial.suggest_int("max_depth", 3, 15),
    #     n_estimators=trial.suggest_int("n_estimators", 100, 1000),
    #     min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
    #     min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
    #     random_state=seed,
    # )
    # return model
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000),
        criterion=trial.suggest_categorical(
            "criterion", ["gini", "entropy", "log_loss"]
        ),
        max_features=trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        max_depth=trial.suggest_int("max_depth", 2, 15),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        random_state=seed,
    )


def suggest_gradient_boosting(
    trial: optuna.Trial, seed: int = SEED
) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, 1000),
        max_depth=trial.suggest_int("max_depth", 2, 15),
        subsample=trial.suggest_float("subsample", 0.2, 1),
        criterion=trial.suggest_categorical(
            "criterion", ["friedman_mse", "squared_error"]
        ),
        max_features=trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        random_state=seed,
    )


def suggest_mlp_classifier(trial: optuna.Trial, seed: int = SEED) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=trial.suggest_categorical(
            "hidden_layer_sizes",
            [
                (256,),
                (256, 256),
                (256, 256, 256),
                (256, 256, 256, 256),
                (512, 512),
                (512, 512, 512),
                (512, 512, 512, 512),
                (1024, 1024),
                (1024, 1024, 1024),
                (1024, 1024, 1024, 1024),
                (2048, 2048),
                (2048, 2048, 2048),
                (2048, 2048, 2048, 2048),
            ],
        ),
        activation=trial.suggest_categorical(
            "activation", ["identity", "logistic", "tanh", "relu"]
        ),
        solver="adam",
        alpha=trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
        learning_rate=trial.suggest_categorical(
            "learning_rate", ["constant", "invscaling", "adaptive"]
        ),
        max_iter=MAX_ITER,
        random_state=seed,
        early_stopping=True,
    )


def suggest_xgboost(
    trial: optuna.Trial, device: str = "cpu", seed: int = SEED
) -> XGBClassifier:
    return XGBClassifier(
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, 1000),
        eval_metric="mlogloss",
        random_state=SEED,
        device=device,
    )


def suggest_lightgbm(trial: optuna.Trial, seed: int = SEED) -> LGBMClassifier:
    return LGBMClassifier(
        max_depth=trial.suggest_int("max_depth", -1, 10),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, 1000),
        random_state=seed,
        lambda_l1=trial.suggest_float("lambda_l1", 0, 1),
        lambda_l2=trial.suggest_float("lambda_l2", 0, 1),
        verbose=-1,
    )


def suggest_catboost(trial: optuna.Trial, seed: int = SEED) -> CatBoostClassifier:
    return CatBoostClassifier(
        depth=trial.suggest_int("depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1, log=True),
        iterations=trial.suggest_int("iterations", 100, 1000),
        verbose=0,
        random_state=seed,
    )
