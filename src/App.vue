<template>
  <div id="app">
    <div id="nav">基于知识图谱的VGG16分类结果语义解释自动生成</div>
    <router-view/>

  </div>


</template>




<style lang="scss">
body {
  margin: 0;
  list-style: none;
  text-decoration: none;
  font-size: 100%;
}
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  font-size:25px;
  font-weight:700;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #030303ef;
}

#nav {
  margin: 5px;
  padding: 20px;

  a {
    font-weight: bold;
    color: #2c3e50;

    &.router-link-exact-active {
      color: #474f4b;
    }
  }
}


</style>
