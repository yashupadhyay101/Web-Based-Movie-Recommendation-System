from django.test import SimpleTestCase
from django.urls import reverse, resolve
from app1.views import home, about, demo, content, content_1, collaboritive, visual, visual2, visual3, showcollaborative, showcontent, showcontent_1, see


class TestsUrls(SimpleTestCase):
    
    def test_demo_url_is_resolved(self):
        url = reverse('demo')
        print(resolve(url))
        self.assertEquals(resolve(url).func, demo)

    def test_content_url_is_resolved(self):
        url = reverse('content')
        print(resolve(url))
        self.assertEquals(resolve(url).func, content)

    def test_content_1_url_is_resolved(self):
        url = reverse('content_1')
        print(resolve(url))
        self.assertEquals(resolve(url).func, content_1)

    def test_collaboritive_url_is_resolved(self):
        url = reverse('collaboritive')
        print(resolve(url))
        self.assertEquals(resolve(url).func, collaboritive)

    def test_visual_url_is_resolved(self):
        url = reverse('visual')
        print(resolve(url))
        self.assertEquals(resolve(url).func, visual)

    def test_visual2_url_is_resolved(self):
        url = reverse('visual2')
        print(resolve(url))
        self.assertEquals(resolve(url).func, visual2)

    def test_visual3_url_is_resolved(self):
        url = reverse('visual3')
        print(resolve(url))
        self.assertEquals(resolve(url).func, visual3)

    def test_showcollaborative_url_is_resolved(self):
        url = reverse('showcollaborative')
        print(resolve(url))
        self.assertEquals(resolve(url).func, showcollaborative)

    def test_showcontent_url_is_resolved(self):
        url = reverse('showcontent')
        print(resolve(url))
        self.assertEquals(resolve(url).func, showcontent)

    def test_showcontent_1_url_is_resolved(self):
        url = reverse('showcontent_1')
        print(resolve(url))
        self.assertEquals(resolve(url).func, showcontent_1)

    def test_see_url_is_resolved(self):
        url = reverse('see')
        print(resolve(url))
        self.assertEquals(resolve(url).func, see)

    def test_about_url_is_resolved(self):
        url = reverse('about')
        print(resolve(url))
        self.assertEquals(resolve(url).func, about)