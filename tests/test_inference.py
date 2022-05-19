from src.infer import infer
from src.persistence import load_models


def test_inference():
    examples = [
        "How to draw a stacked dotplot in R?",
        "mysql select all records where a datetime field is less than a specified value",
        "How to terminate windows phone 8.1 app",
        "get current time in a specific country via jquery",
        "Configuring Tomcat to Use SSL",
    ]
    y_mybag, y_tfidf = infer(examples, *load_models(path='./tests/resources'))
    y_mybag_expected = [('r',), ('mysql', 'php'), ('c#',), ('javascript', 'jquery'), ('java',)]
    y_tfidf_expected = [('r',), ('mysql', 'php'), ('c#',), ('javascript', 'jquery'), ('java',)]
    for y_m, y_m_exp in zip(y_mybag, y_mybag_expected):
        assert y_m == y_m_exp
    for y_t, y_t_exp in zip(y_tfidf, y_tfidf_expected):
        assert y_t == y_t_exp
