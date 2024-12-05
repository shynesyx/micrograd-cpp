#include "engine_unittest.h"
#include <iostream>

TEST(ValueTest, plus) {
  Value a(1.0);
  Value b(2.0);
  Value c = a + b;
  Value d = b - a;

  EXPECT_EQ(c.getData(), 3.0);
  EXPECT_TRUE(c.getOperator() == "+");
  
  std::set<Value*> result = c.getPrev();
  std::set<Value*> expected{&a, &b};

  ASSERT_EQ(result.size(), expected.size());
  EXPECT_THAT(result, ::testing::ContainerEq(expected));

  EXPECT_EQ(d.getData(), 1.0);
  EXPECT_TRUE(d.getOperator() == "-");
 
}

