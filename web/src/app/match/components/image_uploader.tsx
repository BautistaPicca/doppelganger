"use client"
import { useCallback, useRef, useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Upload, X, Camera, ImageIcon } from "lucide-react";

// TODO: Tomar una foto con la cámara del dispositivo no funciona, es dificil de testear en localhost por HTTPS

interface PhotoUploadProps {
  onPhotoSelect: (file: File) => void;
  selectedPhoto: File | null;
  onRemovePhoto: () => void;
}

export const PhotoUpload = ({
  onPhotoSelect,
  selectedPhoto,
  onRemovePhoto,
}: PhotoUploadProps) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const [isCameraActive, setIsCameraActive] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);

  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [stream]);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file && file.type.startsWith("image/")) {
        onPhotoSelect(file);
      }
    },
    [onPhotoSelect]
  );

  const startCamera = useCallback(async () => {
    try {
      const newStream = await navigator.mediaDevices.getUserMedia({ video: true });
      setStream(newStream);
      setIsCameraActive(true);
      if (videoRef.current) {
        videoRef.current.srcObject = newStream;
      }
    } catch (err) {
      console.error("Error al acceder a la cámara:", err);
    }
  }, []);

  const capturePhoto = useCallback(() => {
    if (!videoRef.current) return;
    const video = videoRef.current;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) return;

      const file = new File([blob], "captured.jpg", { type: "image/jpeg" });
      onPhotoSelect(file);

      // Detiene la cámara
      stream?.getTracks().forEach((track) => track.stop());
      setStream(null);
      setIsCameraActive(false);
    }, "image/jpeg");
  }, [onPhotoSelect, stream]);

  if (selectedPhoto) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
      >
        <Card>
          <CardContent className="p-6">
            <div className="relative">
              <div className="aspect-square max-w-sm mx-auto rounded-lg overflow-hidden">
                <img
                  src={URL.createObjectURL(selectedPhoto)}
                  alt="Selected"
                  className="w-full h-full object-cover"
                />
              </div>
              <Button
                variant="destructive"
                size="icon"
                className="absolute -top-2 -right-2 rounded-full"
                onClick={onRemovePhoto}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="text-center mt-4 space-y-2">
              <p className="text-sm font-medium">Imagen lista</p>
              <p className="text-xs text-muted-foreground">
                {selectedPhoto.name} • {(selectedPhoto.size / 1024 / 1024).toFixed(1)}MB
              </p>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  if (isCameraActive) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-center">Tomar una foto</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-center">
          <video
            ref={videoRef}
            autoPlay
            className="mx-auto rounded-lg border shadow-md max-w-sm"
          />
          <div className="flex justify-center gap-3">
            <Button onClick={capturePhoto} className="gap-2">
              <Camera className="h-4 w-4" />
              Capturar foto
            </Button>
            <Button
              variant="outline"
              onClick={() => {
                stream?.getTracks().forEach((t) => t.stop());
                setStream(null);
                setIsCameraActive(false);
              }}
            >
              Cancelar
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-center">Sube tu foto</CardTitle>
      </CardHeader>
      <CardContent>
        <motion.div
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary/50 transition-colors"
        >
          <div className="space-y-4">
            <div className="flex justify-center">
              <div className="p-4 rounded-full bg-primary/10">
                <Upload className="h-8 w-8 text-primary" />
              </div>
            </div>

            <p className="text-sm text-muted-foreground mb-4">
              Toma una selfie o sube una foto de tu rostro
            </p>

            <div className="flex justify-center gap-4">
              <Button
                variant="outline"
                size="sm"
                className="gap-2"
                onClick={() => fileInputRef.current?.click()}
              >
                <ImageIcon className="h-4 w-4" />
                Subir foto
              </Button>
            </div>

            <p className="text-xs text-muted-foreground">Soporta JPG, PNG</p>
          </div>
        </motion.div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
        />
      </CardContent>
    </Card>
  );
};
