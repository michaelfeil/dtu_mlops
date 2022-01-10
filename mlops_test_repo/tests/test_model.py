import pytest
from dtumlops.models.model import MyAwesomeModel
import torch

def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
    
    with pytest.raises(ValueError, match=r'Expected each sample to have shape .*'):
        model(torch.randn(1,2,3,4))

@pytest.mark.parametrize("test_input, expected", [([1,1,28,28], [1,10]), ([2,1,28,28], [2,10])])
def test_model_correct_shape(test_input, expected):
    model = MyAwesomeModel()
    for i in [1,2,3]:
        out = list(model(torch.randn(*test_input)).shape)

        assert out == expected, f"model output shape incorrect {out} expected {expected}"
