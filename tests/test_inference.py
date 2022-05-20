from src.infer import infer
from src.persistence import train_and_save_models, load_models


test_res_dir = './tests/resources'
test_model_path = './tests/resources/model.joblib'


def test_model_train_save():
    train_and_save_models(test_res_dir, test_model_path)


def test_inference():
    examples = [
        "How to draw a stacked dotplot in R?",
        "mysql select all records where a datetime field is less than a specified value",
        "How to terminate windows phone 8.1 app",
        "get current time in a specific country via jquery",
        "Configuring Tomcat to Use SSL",
    ]
    y_mybag, y_tfidf = infer(examples, *load_models(test_model_path))
    print(y_mybag)
    print(y_tfidf)
    y_mybag_expected = [(), (), (), ('javascript', 'jquery'), ()]
    y_tfidf_expected = [(), (), (), ('javascript',), ()]
    for y_m, y_m_exp in zip(y_mybag, y_mybag_expected):
        assert y_m == y_m_exp
    for y_t, y_t_exp in zip(y_tfidf, y_tfidf_expected):
        assert y_t == y_t_exp
