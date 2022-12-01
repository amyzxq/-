<template>
<div id='front' style="margin-top: 20px;width: 500px;">
    <el-upload action="" accept="image/jpeg,image/png" :on-change="onUploadChange" :auto-upload="false" :show-file-list="true"
     :file-list="uploadFiles" multiple :limit="1">
    <el-button class="edit_btn" slot="trigger" size="small" type="primary" round>选取文件</el-button>
    <el-button class="edit_btn" style="margin-left: 10px;" size="small" type="success" @click="submitUpload" round>上传到服务器</el-button>

    <div slot="tip" class="el-upload__tip">只能上传jpg/png文件，且不超过500kb</div>
  
  </el-upload>
</div>

</template>

<script>
import axios from 'axios'
  export default{
    data(){
      return {
        uploadFiles: [],
        input: '',
        mode: '1',
        // 后台请求到的json数据
        data: require('../data/records.json'),
        results: []
      }
    },
    methods:{
      submitUpload(){
        if(this.uploadFiles.length ==1 ){
          this.$message.success('请稍等...');
          // 图片上传成功，开始传递给后台
          var that = this;
          var file = this.uploadFiles[0]
          console.log(file)
          var formData = new FormData();
          formData.append("picname", file.name);
          formData.append("img", file.raw);

          let config = {
            headers:{'Content-Type':'multipart/form-data'},
            enctype: "multipart/form-data",
            processData:false,
            contentType:false,  
          };
          axios.post("http://localhost:8888/upload_image/",formData,config).then(function (response){
            console.log(response)
            if(response.status == 200){
              // that.data = require('../data/result.json')
              that.$emit('getData', response.data.data)
            }else{
              that.$message.error(response.statusText);
            }
          })       
        }else if(this.uploadFiles.length > 1){
          this.$message.error('上传文件数量超过一个');
        }else{
          this.$message.error('请上传文件');
        }
      },
      onUploadChange(file, fileList){
        const isIMAGE = (file.raw.type === 'image/jpeg' || file.raw.type === 'image/png' || file.raw.type === 'image/jpg' );
        const isLt1M = file.size / 1024 / 1024 < 1;
        if (!isIMAGE) {
          this.$message.error('上传文件只能是jpg和png格式!');
          return false;
        }
        if (!isLt1M) {
          this.$message.error('上传文件大小不能超过 1MB!');
          return false;
        }
        this.uploadFiles = fileList
      }
    }
  }
</script>

<style lang='scss' scoped>
.el-select {
width: 120px;
// background-color: #fff;
}
.input-with-select .el-input-group__prepend {
background-color: #6ecbf3;
}
a {
  color:#44cef6;
  font-size:30px;
}

.edit_btn {
  min-height: 80px;
  min-width: 150px;
  font-size:20px;
 
}

.el-upload__tip{
  font-size:25px;
  color:#f6f9fb;
}

</style>