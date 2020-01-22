function [d] = GetDummy(a)

b=table2array(a);
c=categorical(b);
d=dummyvar(c);

end
