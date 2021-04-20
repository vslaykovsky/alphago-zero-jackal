#include <gtest/gtest.h>
#include "../src/jackal/jackal.h"
#include "../src/jackal/ground.h"

#include <sstream>
#include <string>

using namespace std;




TEST(UtilTest, sprite) {
    cv::imwrite("tmp/arrow_ru.png", Sprite::load(SpriteType::ARROW_RU).get_image(128));
}



struct TestCopy  {
    int a;

    TestCopy(): a(0){

    }
    TestCopy(const TestCopy& v): a(v.a) {
        cout << "Copy" << endl;
    }

    TestCopy f() {
        TestCopy b(*this);
        b.a++;
        return b;
    }
};

TEST(UtilTest, TestCopy) {
    TestCopy a;

    TestCopy b(a.f());
    TestCopy c;
    c = a.f();
}