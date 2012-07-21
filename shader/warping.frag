#version 420
in vec4 ProjTexCoord[5];
uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;
uniform sampler2D tex3;
uniform sampler2D tex4;
layout(location = 0) out float meanCost;
 layout(location = 1) out vec4 meanColor;
void main()
{
vec4 projTexColor[5];
bool t[5];
for(int i = 0; i<5; i++)
{t[i] = false;}
vec4 baseColor = vec4(0,0,0,0);
float numOfViews = 0;
if(ProjTexCoord[0].x/ProjTexCoord[0].w > 0 && ProjTexCoord[0].x/ProjTexCoord[0].w <1.0 && ProjTexCoord[0].y/ProjTexCoord[0].w > 0 && ProjTexCoord[0].y/ProjTexCoord[0].w < 1.0 && ProjTexCoord[0].z > 0.0f)
{
 projTexColor[0] = textureProj(tex0, ProjTexCoord[0]);
baseColor = baseColor + projTexColor[0];
numOfViews = numOfViews + 1.0;
t[0] = true;
}
if(ProjTexCoord[1].x/ProjTexCoord[1].w > 0 && ProjTexCoord[1].x/ProjTexCoord[1].w <1.0 && ProjTexCoord[1].y/ProjTexCoord[1].w > 0 && ProjTexCoord[1].y/ProjTexCoord[1].w < 1.0 && ProjTexCoord[1].z > 0.0f)
{
 projTexColor[1] = textureProj(tex1, ProjTexCoord[1]);
baseColor = baseColor + projTexColor[1];
numOfViews = numOfViews + 1.0;
t[1] = true;
}
if(ProjTexCoord[2].x/ProjTexCoord[2].w > 0 && ProjTexCoord[2].x/ProjTexCoord[2].w <1.0 && ProjTexCoord[2].y/ProjTexCoord[2].w > 0 && ProjTexCoord[2].y/ProjTexCoord[2].w < 1.0 && ProjTexCoord[2].z > 0.0f)
{
 projTexColor[2] = textureProj(tex2, ProjTexCoord[2]);
baseColor = baseColor + projTexColor[2];
numOfViews = numOfViews + 1.0;
t[2] = true;
}
if(ProjTexCoord[3].x/ProjTexCoord[3].w > 0 && ProjTexCoord[3].x/ProjTexCoord[3].w <1.0 && ProjTexCoord[3].y/ProjTexCoord[3].w > 0 && ProjTexCoord[3].y/ProjTexCoord[3].w < 1.0 && ProjTexCoord[3].z > 0.0f)
{
 projTexColor[3] = textureProj(tex3, ProjTexCoord[3]);
baseColor = baseColor + projTexColor[3];
numOfViews = numOfViews + 1.0;
t[3] = true;
}
if(ProjTexCoord[4].x/ProjTexCoord[4].w > 0 && ProjTexCoord[4].x/ProjTexCoord[4].w <1.0 && ProjTexCoord[4].y/ProjTexCoord[4].w > 0 && ProjTexCoord[4].y/ProjTexCoord[4].w < 1.0 && ProjTexCoord[4].z > 0.0f)
{
 projTexColor[4] = textureProj(tex4, ProjTexCoord[4]);
baseColor = baseColor + projTexColor[4];
numOfViews = numOfViews + 1.0;
t[4] = true;
}
meanColor = vec4(0,0,0,1.0f);
if(numOfViews <=1){
meanCost = 1000000.0f;
}
else
{
baseColor = baseColor/numOfViews;
meanCost = 0.0f;
for(int i = 0; i<projTexColor.length(); i++)
{
if(t[i] == true){
meanCost = meanCost +  float(pow(projTexColor[i].x - baseColor.x, 2)) + float(pow(projTexColor[i].y - baseColor.y, 2)) + float(pow(projTexColor[i].z - baseColor.z, 2));
meanColor.xyz = meanColor.xyz + projTexColor[i].xyz;
}
}
meanColor.xyz = meanColor.xyz/numOfViews;
}
}

