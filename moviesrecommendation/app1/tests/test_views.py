from django.test import TestCase, Client
from django.urls import reverse

class TestViews(TestCase):

    def test_home_GET(self):
        client = Client()

        response = client.get(reverse('demo'))

        self.assertEquals(response.status_code, 200)
        self.assertTemplateUsed(response, 'app1/demo.html')
