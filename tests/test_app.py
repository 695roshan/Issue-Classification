import pytest
from app import app

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client

def test_predict_missing_title(client):
    """
    Test /api/predict endpoint with missing issue title
    """
    test_input={'title': '', 
                'body': 'App crashes when clicking save button'}
    expected_response={'error':'Please enter the issue title'}
    
    actual_response = client.post('/api/predict', data=test_input)

    assert actual_response.status_code == 200
    assert actual_response.get_json() == expected_response


def test_predict_missing_body(client):
    """
    Test /api/predict endpoint with missing issue body
    """
    test_input={'title': 'App crashes', 
                'body': ''}
    expected_response={'error':'Please enter the issue body'}
    
    actual_response = client.post('/api/predict', data=test_input)
    assert actual_response.status_code == 200
    assert actual_response.get_json() == expected_response

def test_predict_valid_input(client, mocker):
    """Test /api/predict with valid inputs."""
    test_input={'title': 'App crashes', 
                'body': 'App crashes when clicking save button'}
    
    #Mocking the model
    mock_insert_into_db = mocker.patch('app.insert_into_db', return_value=1) # Mock the issue id
    mock_make_prediction = mocker.patch('app.make_prediction', return_value=[0.8, 0.1, 0.1])  # Mock probabilities

    expected_response={'id':1,'label':'bug'}
    
    actual_response = client.post('/api/predict', data=test_input)

    assert actual_response.status_code == 200
    assert actual_response.get_json() == expected_response
    # Ensure the mock functions are called
    mock_insert_into_db.assert_called_once()
    mock_make_prediction.assert_called_once()

def test_correct_missing_id(client):
    """
    Test /api/correct endpoint with a missing issue_id
    """
    test_input={'id': '', 
                'label': 'bug'}
    expected_response = {'error':'Please enter an issue id'}
    
    actual_response = client.post('/api/correct', data=test_input)
    assert actual_response.status_code == 200
    assert actual_response.get_json() == expected_response

def test_correct_nonint_id(client):
    """
    Test /api/correct endpoint with a non-integer issue_id
    """
    test_input={'id': 'one', 
                'label': 'bug'}
    expected_response = {'error':'Please enter a valid numeric issue id'}
    
    actual_response = client.post('/api/correct', data=test_input)
    assert actual_response.status_code == 200
    assert actual_response.get_json() == expected_response

def test_correct_invalid_label(client):
    """
    Test /api/correct endpoint with an invalid label
    """
    test_input={'id': '1', 'label': 'feature'}
    expected_response = {'error':'Please enter a valid corrected label (bug,enhancement,question)'}
    
    actual_response = client.post('/api/correct', data=test_input)
    assert actual_response.status_code == 200
    assert actual_response.get_json() == expected_response

def test_correct_valid_input(client, mocker):
    """Test /api/correct with valid inputs."""
    test_input={
        'id': 1,
        'label': 'enhancement'
    }
    mock_update_issue_in_db = mocker.patch('app.update_issue_in_db', return_value={
        "success": "Label for issue id 1 was changed from bug to enhancement"
    })
    expected_response={'success':"Label for issue id 1 was changed from bug to enhancement"}
    
    actual_response = client.post('/api/correct', data=test_input)

    assert actual_response.status_code == 200
    assert actual_response.get_json()==expected_response
    # Ensure the mock function is called
    mock_update_issue_in_db.assert_called_once()