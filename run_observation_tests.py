import unittest
from tests.test_observation_types import TestObservationTypes

if __name__ == "__main__":
    # Create a test suite with our observation type tests
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestObservationTypes)
    
    # Run the tests
    test_result = unittest.TextTestRunner().run(test_suite)
    
    # Check if all tests passed
    if test_result.wasSuccessful():
        print("✅ All observation type tests passed successfully!")
        print("The fixes for case insensitivity are working correctly.")
    else:
        print("❌ Some observation type tests failed. See details above.") 