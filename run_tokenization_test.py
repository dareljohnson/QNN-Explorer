import unittest
from tests.test_tokenization import TestTokenization

if __name__ == "__main__":
    # Create a test suite with our tokenization tests
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestTokenization)
    
    # Run the tests
    test_result = unittest.TextTestRunner().run(test_suite)
    
    # Check if all tests passed
    if test_result.wasSuccessful():
        print("✅ All tokenization tests passed successfully!")
    else:
        print("❌ Some tokenization tests failed. See details above.") 